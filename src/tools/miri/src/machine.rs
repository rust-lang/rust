//! Global machine state as well as implementation of the interpreter engine
//! `Machine` trait.

use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt;
use std::path::Path;
use std::process;

use rand::rngs::StdRng;
use rand::SeedableRng;

use rustc_ast::ast::Mutability;
use rustc_const_eval::const_eval::CheckAlignment;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
#[allow(unused)]
use rustc_data_structures::static_assert_size;
use rustc_middle::{
    mir,
    ty::{
        self,
        layout::{LayoutCx, LayoutError, LayoutOf, TyAndLayout},
        Instance, Ty, TyCtxt, TypeAndMut,
    },
};
use rustc_span::def_id::{CrateNum, DefId};
use rustc_span::Symbol;
use rustc_target::abi::{Align, Size};
use rustc_target::spec::abi::Abi;

use crate::{
    concurrency::{data_race, weak_memory},
    shims::unix::FileHandler,
    *,
};

/// First real-time signal.
/// `signal(7)` says this must be between 32 and 64 and specifies 34 or 35
/// as typical values.
pub const SIGRTMIN: i32 = 34;

/// Last real-time signal.
/// `signal(7)` says it must be between 32 and 64 and specifies
/// `SIGRTMAX` - `SIGRTMIN` >= 8 (which is the value of `_POSIX_RTSIG_MAX`)
pub const SIGRTMAX: i32 = 42;

/// Extra data stored with each stack frame
pub struct FrameExtra<'tcx> {
    /// Extra data for the Borrow Tracker.
    pub borrow_tracker: Option<borrow_tracker::FrameState>,

    /// If this is Some(), then this is a special "catch unwind" frame (the frame of `try_fn`
    /// called by `try`). When this frame is popped during unwinding a panic,
    /// we stop unwinding, use the `CatchUnwindData` to handle catching.
    pub catch_unwind: Option<CatchUnwindData<'tcx>>,

    /// If `measureme` profiling is enabled, holds timing information
    /// for the start of this frame. When we finish executing this frame,
    /// we use this to register a completed event with `measureme`.
    pub timing: Option<measureme::DetachedTiming>,

    /// Indicates whether a `Frame` is part of a workspace-local crate and is also not
    /// `#[track_caller]`. We compute this once on creation and store the result, as an
    /// optimization.
    /// This is used by `MiriMachine::current_span` and `MiriMachine::caller_span`
    pub is_user_relevant: bool,
}

impl<'tcx> std::fmt::Debug for FrameExtra<'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Omitting `timing`, it does not support `Debug`.
        let FrameExtra { borrow_tracker, catch_unwind, timing: _, is_user_relevant: _ } = self;
        f.debug_struct("FrameData")
            .field("borrow_tracker", borrow_tracker)
            .field("catch_unwind", catch_unwind)
            .finish()
    }
}

impl VisitTags for FrameExtra<'_> {
    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
        let FrameExtra { catch_unwind, borrow_tracker, timing: _, is_user_relevant: _ } = self;

        catch_unwind.visit_tags(visit);
        borrow_tracker.visit_tags(visit);
    }
}

/// Extra memory kinds
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MiriMemoryKind {
    /// `__rust_alloc` memory.
    Rust,
    /// `miri_alloc` memory.
    Miri,
    /// `malloc` memory.
    C,
    /// Windows `HeapAlloc` memory.
    WinHeap,
    /// Memory for args, errno, and other parts of the machine-managed environment.
    /// This memory may leak.
    Machine,
    /// Memory allocated by the runtime (e.g. env vars). Separate from `Machine`
    /// because we clean it up and leak-check it.
    Runtime,
    /// Globals copied from `tcx`.
    /// This memory may leak.
    Global,
    /// Memory for extern statics.
    /// This memory may leak.
    ExternStatic,
    /// Memory for thread-local statics.
    /// This memory may leak.
    Tls,
    /// Memory mapped directly by the program
    Mmap,
}

impl From<MiriMemoryKind> for MemoryKind<MiriMemoryKind> {
    #[inline(always)]
    fn from(kind: MiriMemoryKind) -> MemoryKind<MiriMemoryKind> {
        MemoryKind::Machine(kind)
    }
}

impl MayLeak for MiriMemoryKind {
    #[inline(always)]
    fn may_leak(self) -> bool {
        use self::MiriMemoryKind::*;
        match self {
            Rust | Miri | C | WinHeap | Runtime => false,
            Machine | Global | ExternStatic | Tls | Mmap => true,
        }
    }
}

impl fmt::Display for MiriMemoryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use self::MiriMemoryKind::*;
        match self {
            Rust => write!(f, "Rust heap"),
            Miri => write!(f, "Miri bare-metal heap"),
            C => write!(f, "C heap"),
            WinHeap => write!(f, "Windows heap"),
            Machine => write!(f, "machine-managed memory"),
            Runtime => write!(f, "language runtime memory"),
            Global => write!(f, "global (static or const)"),
            ExternStatic => write!(f, "extern static"),
            Tls => write!(f, "thread-local static"),
            Mmap => write!(f, "mmap"),
        }
    }
}

/// Pointer provenance.
#[derive(Clone, Copy)]
pub enum Provenance {
    Concrete {
        alloc_id: AllocId,
        /// Borrow Tracker tag.
        tag: BorTag,
    },
    Wildcard,
}

// This needs to be `Eq`+`Hash` because the `Machine` trait needs that because validity checking
// *might* be recursive and then it has to track which places have already been visited.
// However, comparing provenance is meaningless, since `Wildcard` might be any provenance -- and of
// course we don't actually do recursive checking.
// We could change `RefTracking` to strip provenance for its `seen` set but that type is generic so that is quite annoying.
// Instead owe add the required instances but make them panic.
impl PartialEq for Provenance {
    fn eq(&self, _other: &Self) -> bool {
        panic!("Provenance must not be compared")
    }
}
impl Eq for Provenance {}
impl std::hash::Hash for Provenance {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {
        panic!("Provenance must not be hashed")
    }
}

/// The "extra" information a pointer has over a regular AllocId.
#[derive(Copy, Clone, PartialEq)]
pub enum ProvenanceExtra {
    Concrete(BorTag),
    Wildcard,
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(Pointer<Provenance>, 24);
// FIXME: this would with in 24bytes but layout optimizations are not smart enough
// #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
//static_assert_size!(Pointer<Option<Provenance>>, 24);
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(Scalar<Provenance>, 32);

impl fmt::Debug for Provenance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provenance::Concrete { alloc_id, tag } => {
                // Forward `alternate` flag to `alloc_id` printing.
                if f.alternate() {
                    write!(f, "[{alloc_id:#?}]")?;
                } else {
                    write!(f, "[{alloc_id:?}]")?;
                }
                // Print Borrow Tracker tag.
                write!(f, "{tag:?}")?;
            }
            Provenance::Wildcard => {
                write!(f, "[wildcard]")?;
            }
        }
        Ok(())
    }
}

impl interpret::Provenance for Provenance {
    /// We use absolute addresses in the `offset` of a `Pointer<Provenance>`.
    const OFFSET_IS_ADDR: bool = true;

    fn get_alloc_id(self) -> Option<AllocId> {
        match self {
            Provenance::Concrete { alloc_id, .. } => Some(alloc_id),
            Provenance::Wildcard => None,
        }
    }

    fn join(left: Option<Self>, right: Option<Self>) -> Option<Self> {
        match (left, right) {
            // If both are the *same* concrete tag, that is the result.
            (
                Some(Provenance::Concrete { alloc_id: left_alloc, tag: left_tag }),
                Some(Provenance::Concrete { alloc_id: right_alloc, tag: right_tag }),
            ) if left_alloc == right_alloc && left_tag == right_tag => left,
            // If one side is a wildcard, the best possible outcome is that it is equal to the other
            // one, and we use that.
            (Some(Provenance::Wildcard), o) | (o, Some(Provenance::Wildcard)) => o,
            // Otherwise, fall back to `None`.
            _ => None,
        }
    }
}

impl fmt::Debug for ProvenanceExtra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProvenanceExtra::Concrete(pid) => write!(f, "{pid:?}"),
            ProvenanceExtra::Wildcard => write!(f, "<wildcard>"),
        }
    }
}

impl ProvenanceExtra {
    pub fn and_then<T>(self, f: impl FnOnce(BorTag) -> Option<T>) -> Option<T> {
        match self {
            ProvenanceExtra::Concrete(pid) => f(pid),
            ProvenanceExtra::Wildcard => None,
        }
    }
}

/// Extra per-allocation data
#[derive(Debug, Clone)]
pub struct AllocExtra<'tcx> {
    /// Global state of the borrow tracker, if enabled.
    pub borrow_tracker: Option<borrow_tracker::AllocState>,
    /// Data race detection via the use of a vector-clock.
    /// This is only added if it is enabled.
    pub data_race: Option<data_race::AllocState>,
    /// Weak memory emulation via the use of store buffers.
    /// This is only added if it is enabled.
    pub weak_memory: Option<weak_memory::AllocState>,
    /// A backtrace to where this allocation was allocated.
    /// As this is recorded for leak reports, it only exists
    /// if this allocation is leakable. The backtrace is not
    /// pruned yet; that should be done before printing it.
    pub backtrace: Option<Vec<FrameInfo<'tcx>>>,
}

impl VisitTags for AllocExtra<'_> {
    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
        let AllocExtra { borrow_tracker, data_race, weak_memory, backtrace: _ } = self;

        borrow_tracker.visit_tags(visit);
        data_race.visit_tags(visit);
        weak_memory.visit_tags(visit);
    }
}

/// Precomputed layouts of primitive types
pub struct PrimitiveLayouts<'tcx> {
    pub unit: TyAndLayout<'tcx>,
    pub i8: TyAndLayout<'tcx>,
    pub i16: TyAndLayout<'tcx>,
    pub i32: TyAndLayout<'tcx>,
    pub i64: TyAndLayout<'tcx>,
    pub i128: TyAndLayout<'tcx>,
    pub isize: TyAndLayout<'tcx>,
    pub u8: TyAndLayout<'tcx>,
    pub u16: TyAndLayout<'tcx>,
    pub u32: TyAndLayout<'tcx>,
    pub u64: TyAndLayout<'tcx>,
    pub u128: TyAndLayout<'tcx>,
    pub usize: TyAndLayout<'tcx>,
    pub bool: TyAndLayout<'tcx>,
    pub mut_raw_ptr: TyAndLayout<'tcx>,   // *mut ()
    pub const_raw_ptr: TyAndLayout<'tcx>, // *const ()
}

impl<'mir, 'tcx: 'mir> PrimitiveLayouts<'tcx> {
    fn new(layout_cx: LayoutCx<'tcx, TyCtxt<'tcx>>) -> Result<Self, &'tcx LayoutError<'tcx>> {
        let tcx = layout_cx.tcx;
        let mut_raw_ptr = tcx.mk_ptr(TypeAndMut { ty: tcx.types.unit, mutbl: Mutability::Mut });
        let const_raw_ptr = tcx.mk_ptr(TypeAndMut { ty: tcx.types.unit, mutbl: Mutability::Not });
        Ok(Self {
            unit: layout_cx.layout_of(tcx.mk_unit())?,
            i8: layout_cx.layout_of(tcx.types.i8)?,
            i16: layout_cx.layout_of(tcx.types.i16)?,
            i32: layout_cx.layout_of(tcx.types.i32)?,
            i64: layout_cx.layout_of(tcx.types.i64)?,
            i128: layout_cx.layout_of(tcx.types.i128)?,
            isize: layout_cx.layout_of(tcx.types.isize)?,
            u8: layout_cx.layout_of(tcx.types.u8)?,
            u16: layout_cx.layout_of(tcx.types.u16)?,
            u32: layout_cx.layout_of(tcx.types.u32)?,
            u64: layout_cx.layout_of(tcx.types.u64)?,
            u128: layout_cx.layout_of(tcx.types.u128)?,
            usize: layout_cx.layout_of(tcx.types.usize)?,
            bool: layout_cx.layout_of(tcx.types.bool)?,
            mut_raw_ptr: layout_cx.layout_of(mut_raw_ptr)?,
            const_raw_ptr: layout_cx.layout_of(const_raw_ptr)?,
        })
    }

    pub fn uint(&self, size: Size) -> Option<TyAndLayout<'tcx>> {
        match size.bits() {
            8 => Some(self.u8),
            16 => Some(self.u16),
            32 => Some(self.u32),
            64 => Some(self.u64),
            128 => Some(self.u128),
            _ => None,
        }
    }

    pub fn int(&self, size: Size) -> Option<TyAndLayout<'tcx>> {
        match size.bits() {
            8 => Some(self.i8),
            16 => Some(self.i16),
            32 => Some(self.i32),
            64 => Some(self.i64),
            128 => Some(self.i128),
            _ => None,
        }
    }
}

/// The machine itself.
///
/// If you add anything here that stores machine values, remember to update
/// `visit_all_machine_values`!
pub struct MiriMachine<'mir, 'tcx> {
    // We carry a copy of the global `TyCtxt` for convenience, so methods taking just `&Evaluator` have `tcx` access.
    pub tcx: TyCtxt<'tcx>,

    /// Global data for borrow tracking.
    pub borrow_tracker: Option<borrow_tracker::GlobalState>,

    /// Data race detector global data.
    pub data_race: Option<data_race::GlobalState>,

    /// Ptr-int-cast module global data.
    pub intptrcast: intptrcast::GlobalState,

    /// Environment variables set by `setenv`.
    /// Miri does not expose env vars from the host to the emulated program.
    pub(crate) env_vars: EnvVars<'tcx>,

    /// Return place of the main function.
    pub(crate) main_fn_ret_place: Option<MemPlace<Provenance>>,

    /// Program arguments (`Option` because we can only initialize them after creating the ecx).
    /// These are *pointers* to argc/argv because macOS.
    /// We also need the full command line as one string because of Windows.
    pub(crate) argc: Option<MemPlace<Provenance>>,
    pub(crate) argv: Option<MemPlace<Provenance>>,
    pub(crate) cmd_line: Option<MemPlace<Provenance>>,

    /// TLS state.
    pub(crate) tls: TlsData<'tcx>,

    /// What should Miri do when an op requires communicating with the host,
    /// such as accessing host env vars, random number generation, and
    /// file system access.
    pub(crate) isolated_op: IsolatedOp,

    /// Whether to enforce the validity invariant.
    pub(crate) validate: bool,

    /// Whether to enforce [ABI](Abi) of function calls.
    pub(crate) enforce_abi: bool,

    /// The table of file descriptors.
    pub(crate) file_handler: shims::unix::FileHandler,
    /// The table of directory descriptors.
    pub(crate) dir_handler: shims::unix::DirHandler,

    /// This machine's monotone clock.
    pub(crate) clock: Clock,

    /// The set of threads.
    pub(crate) threads: ThreadManager<'mir, 'tcx>,

    /// Precomputed `TyLayout`s for primitive data types that are commonly used inside Miri.
    pub(crate) layouts: PrimitiveLayouts<'tcx>,

    /// Allocations that are considered roots of static memory (that may leak).
    pub(crate) static_roots: Vec<AllocId>,

    /// The `measureme` profiler used to record timing information about
    /// the emulated program.
    profiler: Option<measureme::Profiler>,
    /// Used with `profiler` to cache the `StringId`s for event names
    /// uesd with `measureme`.
    string_cache: FxHashMap<String, measureme::StringId>,

    /// Cache of `Instance` exported under the given `Symbol` name.
    /// `None` means no `Instance` exported under the given name is found.
    pub(crate) exported_symbols_cache: FxHashMap<Symbol, Option<Instance<'tcx>>>,

    /// Whether to raise a panic in the context of the evaluated process when unsupported
    /// functionality is encountered. If `false`, an error is propagated in the Miri application context
    /// instead (default behavior)
    pub(crate) panic_on_unsupported: bool,

    /// Equivalent setting as RUST_BACKTRACE on encountering an error.
    pub(crate) backtrace_style: BacktraceStyle,

    /// Crates which are considered local for the purposes of error reporting.
    pub(crate) local_crates: Vec<CrateNum>,

    /// Mapping extern static names to their base pointer.
    extern_statics: FxHashMap<Symbol, Pointer<Provenance>>,

    /// The random number generator used for resolving non-determinism.
    /// Needs to be queried by ptr_to_int, hence needs interior mutability.
    pub(crate) rng: RefCell<StdRng>,

    /// The allocation IDs to report when they are being allocated
    /// (helps for debugging memory leaks and use after free bugs).
    tracked_alloc_ids: FxHashSet<AllocId>,

    /// Controls whether alignment of memory accesses is being checked.
    pub(crate) check_alignment: AlignmentCheck,

    /// Failure rate of compare_exchange_weak, between 0.0 and 1.0
    pub(crate) cmpxchg_weak_failure_rate: f64,

    /// Corresponds to -Zmiri-mute-stdout-stderr and doesn't write the output but acts as if it succeeded.
    pub(crate) mute_stdout_stderr: bool,

    /// Whether weak memory emulation is enabled
    pub(crate) weak_memory: bool,

    /// The probability of the active thread being preempted at the end of each basic block.
    pub(crate) preemption_rate: f64,

    /// If `Some`, we will report the current stack every N basic blocks.
    pub(crate) report_progress: Option<u32>,
    // The total number of blocks that have been executed.
    pub(crate) basic_block_count: u64,

    /// Handle of the optional shared object file for external functions.
    #[cfg(target_os = "linux")]
    pub external_so_lib: Option<(libloading::Library, std::path::PathBuf)>,
    #[cfg(not(target_os = "linux"))]
    pub external_so_lib: Option<!>,

    /// Run a garbage collector for BorTags every N basic blocks.
    pub(crate) gc_interval: u32,
    /// The number of blocks that passed since the last BorTag GC pass.
    pub(crate) since_gc: u32,

    /// The number of CPUs to be reported by miri.
    pub(crate) num_cpus: u32,

    /// Determines Miri's page size and associated values
    pub(crate) page_size: u64,
    pub(crate) stack_addr: u64,
    pub(crate) stack_size: u64,

    /// Whether to collect a backtrace when each allocation is created, just in case it leaks.
    pub(crate) collect_leak_backtraces: bool,
}

impl<'mir, 'tcx> MiriMachine<'mir, 'tcx> {
    pub(crate) fn new(config: &MiriConfig, layout_cx: LayoutCx<'tcx, TyCtxt<'tcx>>) -> Self {
        let tcx = layout_cx.tcx;
        let local_crates = helpers::get_local_crates(tcx);
        let layouts =
            PrimitiveLayouts::new(layout_cx).expect("Couldn't get layouts of primitive types");
        let profiler = config.measureme_out.as_ref().map(|out| {
            let crate_name = layout_cx
                .tcx
                .sess
                .opts
                .crate_name
                .clone()
                .unwrap_or_else(|| "unknown-crate".to_string());
            let pid = process::id();
            // We adopt the same naming scheme for the profiler output that rustc uses. In rustc,
            // the PID is padded so that the nondeterministic value of the PID does not spread
            // nondeterminisim to the allocator. In Miri we are not aiming for such performance
            // control, we just pad for consistency with rustc.
            let filename = format!("{crate_name}-{pid:07}");
            let path = Path::new(out).join(filename);
            measureme::Profiler::new(path).expect("Couldn't create `measureme` profiler")
        });
        let rng = StdRng::seed_from_u64(config.seed.unwrap_or(0));
        let borrow_tracker = config.borrow_tracker.map(|bt| bt.instantiate_global_state(config));
        let data_race = config.data_race_detector.then(|| data_race::GlobalState::new(config));
        // Determine page size, stack address, and stack size.
        // These values are mostly meaningless, but the stack address is also where we start
        // allocating physical integer addresses for all allocations.
        let page_size = if let Some(page_size) = config.page_size {
            page_size
        } else {
            let target = &tcx.sess.target;
            match target.arch.as_ref() {
                "wasm32" | "wasm64" => 64 * 1024, // https://webassembly.github.io/spec/core/exec/runtime.html#memory-instances
                "aarch64" =>
                    if target.options.vendor.as_ref() == "apple" {
                        // No "definitive" source, but see:
                        // https://www.wwdcnotes.com/notes/wwdc20/10214/
                        // https://github.com/ziglang/zig/issues/11308 etc.
                        16 * 1024
                    } else {
                        4 * 1024
                    },
                _ => 4 * 1024,
            }
        };
        // On 16bit targets, 32 pages is more than the entire address space!
        let stack_addr = if tcx.pointer_size().bits() < 32 { page_size } else { page_size * 32 };
        let stack_size =
            if tcx.pointer_size().bits() < 32 { page_size * 4 } else { page_size * 16 };
        MiriMachine {
            tcx,
            borrow_tracker,
            data_race,
            intptrcast: RefCell::new(intptrcast::GlobalStateInner::new(config, stack_addr)),
            // `env_vars` depends on a full interpreter so we cannot properly initialize it yet.
            env_vars: EnvVars::default(),
            main_fn_ret_place: None,
            argc: None,
            argv: None,
            cmd_line: None,
            tls: TlsData::default(),
            isolated_op: config.isolated_op,
            validate: config.validate,
            enforce_abi: config.check_abi,
            file_handler: FileHandler::new(config.mute_stdout_stderr),
            dir_handler: Default::default(),
            layouts,
            threads: ThreadManager::default(),
            static_roots: Vec::new(),
            profiler,
            string_cache: Default::default(),
            exported_symbols_cache: FxHashMap::default(),
            panic_on_unsupported: config.panic_on_unsupported,
            backtrace_style: config.backtrace_style,
            local_crates,
            extern_statics: FxHashMap::default(),
            rng: RefCell::new(rng),
            tracked_alloc_ids: config.tracked_alloc_ids.clone(),
            check_alignment: config.check_alignment,
            cmpxchg_weak_failure_rate: config.cmpxchg_weak_failure_rate,
            mute_stdout_stderr: config.mute_stdout_stderr,
            weak_memory: config.weak_memory_emulation,
            preemption_rate: config.preemption_rate,
            report_progress: config.report_progress,
            basic_block_count: 0,
            clock: Clock::new(config.isolated_op == IsolatedOp::Allow),
            #[cfg(target_os = "linux")]
            external_so_lib: config.external_so_file.as_ref().map(|lib_file_path| {
                let target_triple = layout_cx.tcx.sess.opts.target_triple.triple();
                // Check if host target == the session target.
                if env!("TARGET") != target_triple {
                    panic!(
                        "calling external C functions in linked .so file requires host and target to be the same: host={}, target={}",
                        env!("TARGET"),
                        target_triple,
                    );
                }
                // Note: it is the user's responsibility to provide a correct SO file.
                // WATCH OUT: If an invalid/incorrect SO file is specified, this can cause
                // undefined behaviour in Miri itself!
                (
                    unsafe {
                        libloading::Library::new(lib_file_path)
                            .expect("failed to read specified extern shared object file")
                    },
                    lib_file_path.clone(),
                )
            }),
            #[cfg(not(target_os = "linux"))]
            external_so_lib: config.external_so_file.as_ref().map(|_| {
                panic!("loading external .so files is only supported on Linux")
            }),
            gc_interval: config.gc_interval,
            since_gc: 0,
            num_cpus: config.num_cpus,
            page_size,
            stack_addr,
            stack_size,
            collect_leak_backtraces: config.collect_leak_backtraces,
        }
    }

    pub(crate) fn late_init(
        this: &mut MiriInterpCx<'mir, 'tcx>,
        config: &MiriConfig,
        on_main_stack_empty: StackEmptyCallback<'mir, 'tcx>,
    ) -> InterpResult<'tcx> {
        EnvVars::init(this, config)?;
        MiriMachine::init_extern_statics(this)?;
        ThreadManager::init(this, on_main_stack_empty);
        Ok(())
    }

    fn add_extern_static(
        this: &mut MiriInterpCx<'mir, 'tcx>,
        name: &str,
        ptr: Pointer<Option<Provenance>>,
    ) {
        // This got just allocated, so there definitely is a pointer here.
        let ptr = ptr.into_pointer_or_addr().unwrap();
        this.machine.extern_statics.try_insert(Symbol::intern(name), ptr).unwrap();
    }

    fn alloc_extern_static(
        this: &mut MiriInterpCx<'mir, 'tcx>,
        name: &str,
        val: ImmTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let place = this.allocate(val.layout, MiriMemoryKind::ExternStatic.into())?;
        this.write_immediate(*val, &place.into())?;
        Self::add_extern_static(this, name, place.ptr);
        Ok(())
    }

    /// Sets up the "extern statics" for this machine.
    fn init_extern_statics(this: &mut MiriInterpCx<'mir, 'tcx>) -> InterpResult<'tcx> {
        // "__rust_no_alloc_shim_is_unstable"
        let val = ImmTy::from_int(0, this.machine.layouts.u8);
        Self::alloc_extern_static(this, "__rust_no_alloc_shim_is_unstable", val)?;

        match this.tcx.sess.target.os.as_ref() {
            "linux" => {
                // "environ"
                Self::add_extern_static(
                    this,
                    "environ",
                    this.machine.env_vars.environ.unwrap().ptr,
                );
                // A couple zero-initialized pointer-sized extern statics.
                // Most of them are for weak symbols, which we all set to null (indicating that the
                // symbol is not supported, and triggering fallback code which ends up calling a
                // syscall that we do support).
                for name in &["__cxa_thread_atexit_impl", "getrandom", "statx", "__clock_gettime64"]
                {
                    let val = ImmTy::from_int(0, this.machine.layouts.usize);
                    Self::alloc_extern_static(this, name, val)?;
                }
            }
            "freebsd" => {
                // "environ"
                Self::add_extern_static(
                    this,
                    "environ",
                    this.machine.env_vars.environ.unwrap().ptr,
                );
            }
            "android" => {
                // "signal"
                let layout = this.machine.layouts.const_raw_ptr;
                let dlsym = Dlsym::from_str("signal".as_bytes(), &this.tcx.sess.target.os)?
                    .expect("`signal` must be an actual dlsym on android");
                let ptr = this.create_fn_alloc_ptr(FnVal::Other(dlsym));
                let val = ImmTy::from_scalar(Scalar::from_pointer(ptr, this), layout);
                Self::alloc_extern_static(this, "signal", val)?;
                // A couple zero-initialized pointer-sized extern statics.
                // Most of them are for weak symbols, which we all set to null (indicating that the
                // symbol is not supported, and triggering fallback code.)
                for name in &["bsd_signal"] {
                    let val = ImmTy::from_int(0, this.machine.layouts.usize);
                    Self::alloc_extern_static(this, name, val)?;
                }
            }
            "windows" => {
                // "_tls_used"
                // This is some obscure hack that is part of the Windows TLS story. It's a `u8`.
                let val = ImmTy::from_int(0, this.machine.layouts.u8);
                Self::alloc_extern_static(this, "_tls_used", val)?;
            }
            _ => {} // No "extern statics" supported on this target
        }
        Ok(())
    }

    pub(crate) fn communicate(&self) -> bool {
        self.isolated_op == IsolatedOp::Allow
    }

    /// Check whether the stack frame that this `FrameInfo` refers to is part of a local crate.
    pub(crate) fn is_local(&self, frame: &FrameInfo<'_>) -> bool {
        let def_id = frame.instance.def_id();
        def_id.is_local() || self.local_crates.contains(&def_id.krate)
    }

    /// Called when the interpreter is going to shut down abnormally, such as due to a Ctrl-C.
    pub(crate) fn handle_abnormal_termination(&mut self) {
        // All strings in the profile data are stored in a single string table which is not
        // written to disk until the profiler is dropped. If the interpreter exits without dropping
        // the profiler, it is not possible to interpret the profile data and all measureme tools
        // will panic when given the file.
        drop(self.profiler.take());
    }

    pub(crate) fn round_up_to_multiple_of_page_size(&self, length: u64) -> Option<u64> {
        #[allow(clippy::arithmetic_side_effects)] // page size is nonzero
        (length.checked_add(self.page_size - 1)? / self.page_size).checked_mul(self.page_size)
    }

    pub(crate) fn page_align(&self) -> Align {
        Align::from_bytes(self.page_size).unwrap()
    }
}

impl VisitTags for MiriMachine<'_, '_> {
    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
        #[rustfmt::skip]
        let MiriMachine {
            threads,
            tls,
            env_vars,
            main_fn_ret_place,
            argc,
            argv,
            cmd_line,
            extern_statics,
            dir_handler,
            borrow_tracker,
            data_race,
            intptrcast,
            file_handler,
            tcx: _,
            isolated_op: _,
            validate: _,
            enforce_abi: _,
            clock: _,
            layouts: _,
            static_roots: _,
            profiler: _,
            string_cache: _,
            exported_symbols_cache: _,
            panic_on_unsupported: _,
            backtrace_style: _,
            local_crates: _,
            rng: _,
            tracked_alloc_ids: _,
            check_alignment: _,
            cmpxchg_weak_failure_rate: _,
            mute_stdout_stderr: _,
            weak_memory: _,
            preemption_rate: _,
            report_progress: _,
            basic_block_count: _,
            external_so_lib: _,
            gc_interval: _,
            since_gc: _,
            num_cpus: _,
            page_size: _,
            stack_addr: _,
            stack_size: _,
            collect_leak_backtraces: _,
        } = self;

        threads.visit_tags(visit);
        tls.visit_tags(visit);
        env_vars.visit_tags(visit);
        dir_handler.visit_tags(visit);
        file_handler.visit_tags(visit);
        data_race.visit_tags(visit);
        borrow_tracker.visit_tags(visit);
        intptrcast.visit_tags(visit);
        main_fn_ret_place.visit_tags(visit);
        argc.visit_tags(visit);
        argv.visit_tags(visit);
        cmd_line.visit_tags(visit);
        for ptr in extern_statics.values() {
            ptr.visit_tags(visit);
        }
    }
}

/// A rustc InterpCx for Miri.
pub type MiriInterpCx<'mir, 'tcx> = InterpCx<'mir, 'tcx, MiriMachine<'mir, 'tcx>>;

/// A little trait that's useful to be inherited by extension traits.
pub trait MiriInterpCxExt<'mir, 'tcx> {
    fn eval_context_ref<'a>(&'a self) -> &'a MiriInterpCx<'mir, 'tcx>;
    fn eval_context_mut<'a>(&'a mut self) -> &'a mut MiriInterpCx<'mir, 'tcx>;
}
impl<'mir, 'tcx> MiriInterpCxExt<'mir, 'tcx> for MiriInterpCx<'mir, 'tcx> {
    #[inline(always)]
    fn eval_context_ref(&self) -> &MiriInterpCx<'mir, 'tcx> {
        self
    }
    #[inline(always)]
    fn eval_context_mut(&mut self) -> &mut MiriInterpCx<'mir, 'tcx> {
        self
    }
}

/// Machine hook implementations.
impl<'mir, 'tcx> Machine<'mir, 'tcx> for MiriMachine<'mir, 'tcx> {
    type MemoryKind = MiriMemoryKind;
    type ExtraFnVal = Dlsym;

    type FrameExtra = FrameExtra<'tcx>;
    type AllocExtra = AllocExtra<'tcx>;

    type Provenance = Provenance;
    type ProvenanceExtra = ProvenanceExtra;
    type Bytes = Box<[u8]>;

    type MemoryMap = MonoHashMap<
        AllocId,
        (MemoryKind<MiriMemoryKind>, Allocation<Provenance, Self::AllocExtra, Self::Bytes>),
    >;

    const GLOBAL_KIND: Option<MiriMemoryKind> = Some(MiriMemoryKind::Global);

    const PANIC_ON_ALLOC_FAIL: bool = false;

    #[inline(always)]
    fn enforce_alignment(ecx: &MiriInterpCx<'mir, 'tcx>) -> CheckAlignment {
        if ecx.machine.check_alignment == AlignmentCheck::None {
            CheckAlignment::No
        } else {
            CheckAlignment::Error
        }
    }

    #[inline(always)]
    fn use_addr_for_alignment_check(ecx: &MiriInterpCx<'mir, 'tcx>) -> bool {
        ecx.machine.check_alignment == AlignmentCheck::Int
    }

    fn alignment_check_failed(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        has: Align,
        required: Align,
        _check: CheckAlignment,
    ) -> InterpResult<'tcx, ()> {
        throw_ub!(AlignmentCheckFailed { has, required })
    }

    #[inline(always)]
    fn enforce_validity(ecx: &MiriInterpCx<'mir, 'tcx>, _layout: TyAndLayout<'tcx>) -> bool {
        ecx.machine.validate
    }

    #[inline(always)]
    fn enforce_abi(ecx: &MiriInterpCx<'mir, 'tcx>) -> bool {
        ecx.machine.enforce_abi
    }

    #[inline(always)]
    fn ignore_optional_overflow_checks(ecx: &MiriInterpCx<'mir, 'tcx>) -> bool {
        !ecx.tcx.sess.overflow_checks()
    }

    #[inline(always)]
    fn find_mir_or_eval_fn(
        ecx: &mut MiriInterpCx<'mir, 'tcx>,
        instance: ty::Instance<'tcx>,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        ret: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx, Option<(&'mir mir::Body<'tcx>, ty::Instance<'tcx>)>> {
        ecx.find_mir_or_eval_fn(instance, abi, args, dest, ret, unwind)
    }

    #[inline(always)]
    fn call_extra_fn(
        ecx: &mut MiriInterpCx<'mir, 'tcx>,
        fn_val: Dlsym,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        ret: Option<mir::BasicBlock>,
        _unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        ecx.call_dlsym(fn_val, abi, args, dest, ret)
    }

    #[inline(always)]
    fn call_intrinsic(
        ecx: &mut MiriInterpCx<'mir, 'tcx>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        ret: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        ecx.call_intrinsic(instance, args, dest, ret, unwind)
    }

    #[inline(always)]
    fn assert_panic(
        ecx: &mut MiriInterpCx<'mir, 'tcx>,
        msg: &mir::AssertMessage<'tcx>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        ecx.assert_panic(msg, unwind)
    }

    #[inline(always)]
    fn abort(_ecx: &mut MiriInterpCx<'mir, 'tcx>, msg: String) -> InterpResult<'tcx, !> {
        throw_machine_stop!(TerminationInfo::Abort(msg))
    }

    #[inline(always)]
    fn binary_ptr_op(
        ecx: &MiriInterpCx<'mir, 'tcx>,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, Provenance>,
        right: &ImmTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, (Scalar<Provenance>, bool, Ty<'tcx>)> {
        ecx.binary_ptr_op(bin_op, left, right)
    }

    fn thread_local_static_base_pointer(
        ecx: &mut MiriInterpCx<'mir, 'tcx>,
        def_id: DefId,
    ) -> InterpResult<'tcx, Pointer<Provenance>> {
        ecx.get_or_create_thread_local_alloc(def_id)
    }

    fn extern_static_base_pointer(
        ecx: &MiriInterpCx<'mir, 'tcx>,
        def_id: DefId,
    ) -> InterpResult<'tcx, Pointer<Provenance>> {
        let link_name = ecx.item_link_name(def_id);
        if let Some(&ptr) = ecx.machine.extern_statics.get(&link_name) {
            // Various parts of the engine rely on `get_alloc_info` for size and alignment
            // information. That uses the type information of this static.
            // Make sure it matches the Miri allocation for this.
            let Provenance::Concrete { alloc_id, .. } = ptr.provenance else {
                panic!("extern_statics cannot contain wildcards")
            };
            let (shim_size, shim_align, _kind) = ecx.get_alloc_info(alloc_id);
            let def_ty = ecx.tcx.type_of(def_id).subst_identity();
            let extern_decl_layout = ecx.tcx.layout_of(ty::ParamEnv::empty().and(def_ty)).unwrap();
            if extern_decl_layout.size != shim_size || extern_decl_layout.align.abi != shim_align {
                throw_unsup_format!(
                    "`extern` static `{name}` from crate `{krate}` has been declared \
                    with a size of {decl_size} bytes and alignment of {decl_align} bytes, \
                    but Miri emulates it via an extern static shim \
                    with a size of {shim_size} bytes and alignment of {shim_align} bytes",
                    name = ecx.tcx.def_path_str(def_id),
                    krate = ecx.tcx.crate_name(def_id.krate),
                    decl_size = extern_decl_layout.size.bytes(),
                    decl_align = extern_decl_layout.align.abi.bytes(),
                    shim_size = shim_size.bytes(),
                    shim_align = shim_align.bytes(),
                )
            }
            Ok(ptr)
        } else {
            throw_unsup_format!(
                "`extern` static `{name}` from crate `{krate}` is not supported by Miri",
                name = ecx.tcx.def_path_str(def_id),
                krate = ecx.tcx.crate_name(def_id.krate),
            )
        }
    }

    fn adjust_allocation<'b>(
        ecx: &MiriInterpCx<'mir, 'tcx>,
        id: AllocId,
        alloc: Cow<'b, Allocation>,
        kind: Option<MemoryKind<Self::MemoryKind>>,
    ) -> InterpResult<'tcx, Cow<'b, Allocation<Self::Provenance, Self::AllocExtra>>> {
        let kind = kind.expect("we set our STATIC_KIND so this cannot be None");
        if ecx.machine.tracked_alloc_ids.contains(&id) {
            ecx.emit_diagnostic(NonHaltingDiagnostic::CreatedAlloc(
                id,
                alloc.size(),
                alloc.align,
                kind,
            ));
        }

        let alloc = alloc.into_owned();
        let borrow_tracker = ecx
            .machine
            .borrow_tracker
            .as_ref()
            .map(|bt| bt.borrow_mut().new_allocation(id, alloc.size(), kind, &ecx.machine));

        let race_alloc = ecx.machine.data_race.as_ref().map(|data_race| {
            data_race::AllocState::new_allocation(
                data_race,
                &ecx.machine.threads,
                alloc.size(),
                kind,
                ecx.machine.current_span(),
            )
        });
        let buffer_alloc = ecx.machine.weak_memory.then(weak_memory::AllocState::new_allocation);

        // If an allocation is leaked, we want to report a backtrace to indicate where it was
        // allocated. We don't need to record a backtrace for allocations which are allowed to
        // leak.
        let backtrace = if kind.may_leak() || !ecx.machine.collect_leak_backtraces {
            None
        } else {
            Some(ecx.generate_stacktrace())
        };

        let alloc: Allocation<Provenance, Self::AllocExtra> = alloc.adjust_from_tcx(
            &ecx.tcx,
            AllocExtra {
                borrow_tracker,
                data_race: race_alloc,
                weak_memory: buffer_alloc,
                backtrace,
            },
            |ptr| ecx.global_base_pointer(ptr),
        )?;
        Ok(Cow::Owned(alloc))
    }

    fn adjust_alloc_base_pointer(
        ecx: &MiriInterpCx<'mir, 'tcx>,
        ptr: Pointer<AllocId>,
    ) -> InterpResult<'tcx, Pointer<Provenance>> {
        if cfg!(debug_assertions) {
            // The machine promises to never call us on thread-local or extern statics.
            let alloc_id = ptr.provenance;
            match ecx.tcx.try_get_global_alloc(alloc_id) {
                Some(GlobalAlloc::Static(def_id)) if ecx.tcx.is_thread_local_static(def_id) => {
                    panic!("adjust_alloc_base_pointer called on thread-local static")
                }
                Some(GlobalAlloc::Static(def_id)) if ecx.tcx.is_foreign_item(def_id) => {
                    panic!("adjust_alloc_base_pointer called on extern static")
                }
                _ => {}
            }
        }
        let absolute_addr = intptrcast::GlobalStateInner::rel_ptr_to_addr(ecx, ptr)?;
        let tag = if let Some(borrow_tracker) = &ecx.machine.borrow_tracker {
            borrow_tracker.borrow_mut().base_ptr_tag(ptr.provenance, &ecx.machine)
        } else {
            // Value does not matter, SB is disabled
            BorTag::default()
        };
        Ok(Pointer::new(
            Provenance::Concrete { alloc_id: ptr.provenance, tag },
            Size::from_bytes(absolute_addr),
        ))
    }

    #[inline(always)]
    fn ptr_from_addr_cast(
        ecx: &MiriInterpCx<'mir, 'tcx>,
        addr: u64,
    ) -> InterpResult<'tcx, Pointer<Option<Self::Provenance>>> {
        intptrcast::GlobalStateInner::ptr_from_addr_cast(ecx, addr)
    }

    fn expose_ptr(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        ptr: Pointer<Self::Provenance>,
    ) -> InterpResult<'tcx> {
        match ptr.provenance {
            Provenance::Concrete { alloc_id, tag } =>
                intptrcast::GlobalStateInner::expose_ptr(ecx, alloc_id, tag),
            Provenance::Wildcard => {
                // No need to do anything for wildcard pointers as
                // their provenances have already been previously exposed.
                Ok(())
            }
        }
    }

    /// Convert a pointer with provenance into an allocation-offset pair,
    /// or a `None` with an absolute address if that conversion is not possible.
    fn ptr_get_alloc(
        ecx: &MiriInterpCx<'mir, 'tcx>,
        ptr: Pointer<Self::Provenance>,
    ) -> Option<(AllocId, Size, Self::ProvenanceExtra)> {
        let rel = intptrcast::GlobalStateInner::abs_ptr_to_rel(ecx, ptr);

        rel.map(|(alloc_id, size)| {
            let tag = match ptr.provenance {
                Provenance::Concrete { tag, .. } => ProvenanceExtra::Concrete(tag),
                Provenance::Wildcard => ProvenanceExtra::Wildcard,
            };
            (alloc_id, size, tag)
        })
    }

    #[inline(always)]
    fn before_memory_read(
        _tcx: TyCtxt<'tcx>,
        machine: &Self,
        alloc_extra: &AllocExtra<'tcx>,
        (alloc_id, prov_extra): (AllocId, Self::ProvenanceExtra),
        range: AllocRange,
    ) -> InterpResult<'tcx> {
        if let Some(data_race) = &alloc_extra.data_race {
            data_race.read(alloc_id, range, machine)?;
        }
        if let Some(borrow_tracker) = &alloc_extra.borrow_tracker {
            borrow_tracker.before_memory_read(alloc_id, prov_extra, range, machine)?;
        }
        if let Some(weak_memory) = &alloc_extra.weak_memory {
            weak_memory.memory_accessed(range, machine.data_race.as_ref().unwrap());
        }
        Ok(())
    }

    #[inline(always)]
    fn before_memory_write(
        _tcx: TyCtxt<'tcx>,
        machine: &mut Self,
        alloc_extra: &mut AllocExtra<'tcx>,
        (alloc_id, prov_extra): (AllocId, Self::ProvenanceExtra),
        range: AllocRange,
    ) -> InterpResult<'tcx> {
        if let Some(data_race) = &mut alloc_extra.data_race {
            data_race.write(alloc_id, range, machine)?;
        }
        if let Some(borrow_tracker) = &mut alloc_extra.borrow_tracker {
            borrow_tracker.before_memory_write(alloc_id, prov_extra, range, machine)?;
        }
        if let Some(weak_memory) = &alloc_extra.weak_memory {
            weak_memory.memory_accessed(range, machine.data_race.as_ref().unwrap());
        }
        Ok(())
    }

    #[inline(always)]
    fn before_memory_deallocation(
        _tcx: TyCtxt<'tcx>,
        machine: &mut Self,
        alloc_extra: &mut AllocExtra<'tcx>,
        (alloc_id, prove_extra): (AllocId, Self::ProvenanceExtra),
        range: AllocRange,
    ) -> InterpResult<'tcx> {
        if machine.tracked_alloc_ids.contains(&alloc_id) {
            machine.emit_diagnostic(NonHaltingDiagnostic::FreedAlloc(alloc_id));
        }
        if let Some(data_race) = &mut alloc_extra.data_race {
            data_race.deallocate(alloc_id, range, machine)?;
        }
        if let Some(borrow_tracker) = &mut alloc_extra.borrow_tracker {
            borrow_tracker.before_memory_deallocation(alloc_id, prove_extra, range, machine)?;
        }
        Ok(())
    }

    #[inline(always)]
    fn retag_ptr_value(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        kind: mir::RetagKind,
        val: &ImmTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, Provenance>> {
        if ecx.machine.borrow_tracker.is_some() {
            ecx.retag_ptr_value(kind, val)
        } else {
            Ok(val.clone())
        }
    }

    #[inline(always)]
    fn retag_place_contents(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        kind: mir::RetagKind,
        place: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        if ecx.machine.borrow_tracker.is_some() {
            ecx.retag_place_contents(kind, place)?;
        }
        Ok(())
    }

    #[inline(always)]
    fn init_frame_extra(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        frame: Frame<'mir, 'tcx, Provenance>,
    ) -> InterpResult<'tcx, Frame<'mir, 'tcx, Provenance, FrameExtra<'tcx>>> {
        // Start recording our event before doing anything else
        let timing = if let Some(profiler) = ecx.machine.profiler.as_ref() {
            let fn_name = frame.instance.to_string();
            let entry = ecx.machine.string_cache.entry(fn_name.clone());
            let name = entry.or_insert_with(|| profiler.alloc_string(&*fn_name));

            Some(profiler.start_recording_interval_event_detached(
                *name,
                measureme::EventId::from_label(*name),
                ecx.get_active_thread().to_u32(),
            ))
        } else {
            None
        };

        let borrow_tracker = ecx.machine.borrow_tracker.as_ref();

        let extra = FrameExtra {
            borrow_tracker: borrow_tracker.map(|bt| bt.borrow_mut().new_frame(&ecx.machine)),
            catch_unwind: None,
            timing,
            is_user_relevant: ecx.machine.is_user_relevant(&frame),
        };

        Ok(frame.with_extra(extra))
    }

    fn stack<'a>(
        ecx: &'a InterpCx<'mir, 'tcx, Self>,
    ) -> &'a [Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>] {
        ecx.active_thread_stack()
    }

    fn stack_mut<'a>(
        ecx: &'a mut InterpCx<'mir, 'tcx, Self>,
    ) -> &'a mut Vec<Frame<'mir, 'tcx, Self::Provenance, Self::FrameExtra>> {
        ecx.active_thread_stack_mut()
    }

    fn before_terminator(ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        ecx.machine.basic_block_count += 1u64; // a u64 that is only incremented by 1 will "never" overflow
        ecx.machine.since_gc += 1;
        // Possibly report our progress.
        if let Some(report_progress) = ecx.machine.report_progress {
            if ecx.machine.basic_block_count % u64::from(report_progress) == 0 {
                ecx.emit_diagnostic(NonHaltingDiagnostic::ProgressReport {
                    block_count: ecx.machine.basic_block_count,
                });
            }
        }

        // Search for BorTags to find all live pointers, then remove all other tags from borrow
        // stacks.
        // When debug assertions are enabled, run the GC as often as possible so that any cases
        // where it mistakenly removes an important tag become visible.
        if ecx.machine.gc_interval > 0 && ecx.machine.since_gc >= ecx.machine.gc_interval {
            ecx.machine.since_gc = 0;
            ecx.garbage_collect_tags()?;
        }

        // These are our preemption points.
        ecx.maybe_preempt_active_thread();

        // Make sure some time passes.
        ecx.machine.clock.tick();

        Ok(())
    }

    #[inline(always)]
    fn after_stack_push(ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        if ecx.frame().extra.is_user_relevant {
            // We just pushed a local frame, so we know that the topmost local frame is the topmost
            // frame. If we push a non-local frame, there's no need to do anything.
            let stack_len = ecx.active_thread_stack().len();
            ecx.active_thread_mut().set_top_user_relevant_frame(stack_len - 1);
        }
        if ecx.machine.borrow_tracker.is_some() {
            ecx.retag_return_place()?;
        }
        Ok(())
    }

    #[inline(always)]
    fn after_stack_pop(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        mut frame: Frame<'mir, 'tcx, Provenance, FrameExtra<'tcx>>,
        unwinding: bool,
    ) -> InterpResult<'tcx, StackPopJump> {
        if frame.extra.is_user_relevant {
            // All that we store is whether or not the frame we just removed is local, so now we
            // have no idea where the next topmost local frame is. So we recompute it.
            // (If this ever becomes a bottleneck, we could have `push` store the previous
            // user-relevant frame and restore that here.)
            ecx.active_thread_mut().recompute_top_user_relevant_frame();
        }
        let timing = frame.extra.timing.take();
        if let Some(borrow_tracker) = &ecx.machine.borrow_tracker {
            borrow_tracker.borrow_mut().end_call(&frame.extra);
        }
        let res = ecx.handle_stack_pop_unwind(frame.extra, unwinding);
        if let Some(profiler) = ecx.machine.profiler.as_ref() {
            profiler.finish_recording_interval_event(timing.unwrap());
        }
        res
    }
}
