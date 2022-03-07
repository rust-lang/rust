//! Global machine state as well as implementation of the interpreter engine
//! `Machine` trait.

use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt;
use std::num::NonZeroU64;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::{
    mir,
    ty::{
        self,
        layout::{LayoutCx, LayoutError, LayoutOf, TyAndLayout},
        Instance, TyCtxt,
    },
};
use rustc_span::def_id::DefId;
use rustc_span::symbol::{sym, Symbol};
use rustc_target::abi::Size;
use rustc_target::spec::abi::Abi;

use crate::*;

// Some global facts about the emulated machine.
pub const PAGE_SIZE: u64 = 4 * 1024; // FIXME: adjust to target architecture
pub const STACK_ADDR: u64 = 32 * PAGE_SIZE; // not really about the "stack", but where we start assigning integer addresses to allocations
pub const STACK_SIZE: u64 = 16 * PAGE_SIZE; // whatever
pub const NUM_CPUS: u64 = 1;

/// Extra data stored with each stack frame
pub struct FrameData<'tcx> {
    /// Extra data for Stacked Borrows.
    pub call_id: stacked_borrows::CallId,

    /// If this is Some(), then this is a special "catch unwind" frame (the frame of `try_fn`
    /// called by `try`). When this frame is popped during unwinding a panic,
    /// we stop unwinding, use the `CatchUnwindData` to handle catching.
    pub catch_unwind: Option<CatchUnwindData<'tcx>>,

    /// If `measureme` profiling is enabled, holds timing information
    /// for the start of this frame. When we finish executing this frame,
    /// we use this to register a completed event with `measureme`.
    pub timing: Option<measureme::DetachedTiming>,
}

impl<'tcx> std::fmt::Debug for FrameData<'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Omitting `timing`, it does not support `Debug`.
        let FrameData { call_id, catch_unwind, timing: _ } = self;
        f.debug_struct("FrameData")
            .field("call_id", call_id)
            .field("catch_unwind", catch_unwind)
            .finish()
    }
}

/// Extra memory kinds
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MiriMemoryKind {
    /// `__rust_alloc` memory.
    Rust,
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
}

impl Into<MemoryKind<MiriMemoryKind>> for MiriMemoryKind {
    #[inline(always)]
    fn into(self) -> MemoryKind<MiriMemoryKind> {
        MemoryKind::Machine(self)
    }
}

impl MayLeak for MiriMemoryKind {
    #[inline(always)]
    fn may_leak(self) -> bool {
        use self::MiriMemoryKind::*;
        match self {
            Rust | C | WinHeap | Runtime => false,
            Machine | Global | ExternStatic | Tls => true,
        }
    }
}

impl fmt::Display for MiriMemoryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use self::MiriMemoryKind::*;
        match self {
            Rust => write!(f, "Rust heap"),
            C => write!(f, "C heap"),
            WinHeap => write!(f, "Windows heap"),
            Machine => write!(f, "machine-managed memory"),
            Runtime => write!(f, "language runtime memory"),
            Global => write!(f, "global (static or const)"),
            ExternStatic => write!(f, "extern static"),
            Tls => write!(f, "thread-local static"),
        }
    }
}

/// Pointer provenance (tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tag {
    pub alloc_id: AllocId,
    /// Stacked Borrows tag.
    pub sb: SbTag,
}

impl Provenance for Tag {
    /// We use absolute addresses in the `offset` of a `Pointer<Tag>`.
    const OFFSET_IS_ADDR: bool = true;

    /// We cannot err on partial overwrites, it happens too often in practice (due to unions).
    const ERR_ON_PARTIAL_PTR_OVERWRITE: bool = false;

    fn fmt(ptr: &Pointer<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (tag, addr) = ptr.into_parts(); // address is absolute
        write!(f, "0x{:x}", addr.bytes())?;
        // Forward `alternate` flag to `alloc_id` printing.
        if f.alternate() {
            write!(f, "[{:#?}]", tag.alloc_id)?;
        } else {
            write!(f, "[{:?}]", tag.alloc_id)?;
        }
        // Print Stacked Borrows tag.
        write!(f, "{:?}", tag.sb)
    }

    fn get_alloc_id(self) -> AllocId {
        self.alloc_id
    }
}

/// Extra per-allocation data
#[derive(Debug, Clone)]
pub struct AllocExtra {
    /// Stacked Borrows state is only added if it is enabled.
    pub stacked_borrows: Option<stacked_borrows::AllocExtra>,
    /// Data race detection via the use of a vector-clock,
    ///  this is only added if it is enabled.
    pub data_race: Option<data_race::AllocExtra>,
}

/// Extra global memory data
#[derive(Debug)]
pub struct MemoryExtra {
    pub stacked_borrows: Option<stacked_borrows::MemoryExtra>,
    pub data_race: Option<data_race::MemoryExtra>,
    pub intptrcast: intptrcast::MemoryExtra,

    /// Mapping extern static names to their base pointer.
    extern_statics: FxHashMap<Symbol, Pointer<Tag>>,

    /// The random number generator used for resolving non-determinism.
    /// Needs to be queried by ptr_to_int, hence needs interior mutability.
    pub(crate) rng: RefCell<StdRng>,

    /// An allocation ID to report when it is being allocated
    /// (helps for debugging memory leaks and use after free bugs).
    tracked_alloc_id: Option<AllocId>,

    /// Controls whether alignment of memory accesses is being checked.
    pub(crate) check_alignment: AlignmentCheck,

    /// Failure rate of compare_exchange_weak, between 0.0 and 1.0
    pub(crate) cmpxchg_weak_failure_rate: f64,
}

impl MemoryExtra {
    pub fn new(config: &MiriConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed.unwrap_or(0));
        let stacked_borrows = if config.stacked_borrows {
            Some(RefCell::new(stacked_borrows::GlobalState::new(
                config.tracked_pointer_tag,
                config.tracked_call_id,
                config.tag_raw,
            )))
        } else {
            None
        };
        let data_race =
            if config.data_race_detector { Some(data_race::GlobalState::new()) } else { None };
        MemoryExtra {
            stacked_borrows,
            data_race,
            intptrcast: Default::default(),
            extern_statics: FxHashMap::default(),
            rng: RefCell::new(rng),
            tracked_alloc_id: config.tracked_alloc_id,
            check_alignment: config.check_alignment,
            cmpxchg_weak_failure_rate: config.cmpxchg_weak_failure_rate,
        }
    }

    fn add_extern_static<'tcx, 'mir>(
        this: &mut MiriEvalContext<'mir, 'tcx>,
        name: &str,
        ptr: Pointer<Option<Tag>>,
    ) {
        let ptr = ptr.into_pointer_or_addr().unwrap();
        this.memory.extra.extern_statics.try_insert(Symbol::intern(name), ptr).unwrap();
    }

    /// Sets up the "extern statics" for this machine.
    pub fn init_extern_statics<'tcx, 'mir>(
        this: &mut MiriEvalContext<'mir, 'tcx>,
    ) -> InterpResult<'tcx> {
        match this.tcx.sess.target.os.as_str() {
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
                for name in &["__cxa_thread_atexit_impl", "getrandom", "statx"] {
                    let layout = this.machine.layouts.usize;
                    let place = this.allocate(layout, MiriMemoryKind::ExternStatic.into())?;
                    this.write_scalar(Scalar::from_machine_usize(0, this), &place.into())?;
                    Self::add_extern_static(this, name, place.ptr);
                }
            }
            "windows" => {
                // "_tls_used"
                // This is some obscure hack that is part of the Windows TLS story. It's a `u8`.
                let layout = this.machine.layouts.u8;
                let place = this.allocate(layout, MiriMemoryKind::ExternStatic.into())?;
                this.write_scalar(Scalar::from_u8(0), &place.into())?;
                Self::add_extern_static(this, "_tls_used", place.ptr);
            }
            _ => {} // No "extern statics" supported on this target
        }
        Ok(())
    }
}

/// Precomputed layouts of primitive types
pub struct PrimitiveLayouts<'tcx> {
    pub unit: TyAndLayout<'tcx>,
    pub i8: TyAndLayout<'tcx>,
    pub i32: TyAndLayout<'tcx>,
    pub isize: TyAndLayout<'tcx>,
    pub u8: TyAndLayout<'tcx>,
    pub u32: TyAndLayout<'tcx>,
    pub usize: TyAndLayout<'tcx>,
    pub bool: TyAndLayout<'tcx>,
}

impl<'mir, 'tcx: 'mir> PrimitiveLayouts<'tcx> {
    fn new(layout_cx: LayoutCx<'tcx, TyCtxt<'tcx>>) -> Result<Self, LayoutError<'tcx>> {
        Ok(Self {
            unit: layout_cx.layout_of(layout_cx.tcx.mk_unit())?,
            i8: layout_cx.layout_of(layout_cx.tcx.types.i8)?,
            i32: layout_cx.layout_of(layout_cx.tcx.types.i32)?,
            isize: layout_cx.layout_of(layout_cx.tcx.types.isize)?,
            u8: layout_cx.layout_of(layout_cx.tcx.types.u8)?,
            u32: layout_cx.layout_of(layout_cx.tcx.types.u32)?,
            usize: layout_cx.layout_of(layout_cx.tcx.types.usize)?,
            bool: layout_cx.layout_of(layout_cx.tcx.types.bool)?,
        })
    }
}

/// The machine itself.
pub struct Evaluator<'mir, 'tcx> {
    /// Environment variables set by `setenv`.
    /// Miri does not expose env vars from the host to the emulated program.
    pub(crate) env_vars: EnvVars<'tcx>,

    /// Program arguments (`Option` because we can only initialize them after creating the ecx).
    /// These are *pointers* to argc/argv because macOS.
    /// We also need the full command line as one string because of Windows.
    pub(crate) argc: Option<MemPlace<Tag>>,
    pub(crate) argv: Option<MemPlace<Tag>>,
    pub(crate) cmd_line: Option<MemPlace<Tag>>,

    /// TLS state.
    pub(crate) tls: TlsData<'tcx>,

    /// What should Miri do when an op requires communicating with the host,
    /// such as accessing host env vars, random number generation, and
    /// file system access.
    pub(crate) isolated_op: IsolatedOp,

    /// Whether to enforce the validity invariant.
    pub(crate) validate: bool,

    /// Whether to enforce validity (e.g., initialization) of integers and floats.
    pub(crate) enforce_number_validity: bool,

    /// Whether to enforce [ABI](Abi) of function calls.
    pub(crate) enforce_abi: bool,

    pub(crate) file_handler: shims::posix::FileHandler,
    pub(crate) dir_handler: shims::posix::DirHandler,

    /// The "time anchor" for this machine's monotone clock (for `Instant` simulation).
    pub(crate) time_anchor: Instant,

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
}

impl<'mir, 'tcx> Evaluator<'mir, 'tcx> {
    pub(crate) fn new(config: &MiriConfig, layout_cx: LayoutCx<'tcx, TyCtxt<'tcx>>) -> Self {
        let layouts =
            PrimitiveLayouts::new(layout_cx).expect("Couldn't get layouts of primitive types");
        let profiler = config.measureme_out.as_ref().map(|out| {
            measureme::Profiler::new(out).expect("Couldn't create `measureme` profiler")
        });
        Evaluator {
            // `env_vars` could be initialized properly here if `Memory` were available before
            // calling this method.
            env_vars: EnvVars::default(),
            argc: None,
            argv: None,
            cmd_line: None,
            tls: TlsData::default(),
            isolated_op: config.isolated_op,
            validate: config.validate,
            enforce_number_validity: config.check_number_validity,
            enforce_abi: config.check_abi,
            file_handler: Default::default(),
            dir_handler: Default::default(),
            time_anchor: Instant::now(),
            layouts,
            threads: ThreadManager::default(),
            static_roots: Vec::new(),
            profiler,
            string_cache: Default::default(),
            exported_symbols_cache: FxHashMap::default(),
            panic_on_unsupported: config.panic_on_unsupported,
            backtrace_style: config.backtrace_style,
        }
    }

    pub(crate) fn communicate(&self) -> bool {
        self.isolated_op == IsolatedOp::Allow
    }
}

/// A rustc InterpCx for Miri.
pub type MiriEvalContext<'mir, 'tcx> = InterpCx<'mir, 'tcx, Evaluator<'mir, 'tcx>>;

/// A little trait that's useful to be inherited by extension traits.
pub trait MiriEvalContextExt<'mir, 'tcx> {
    fn eval_context_ref<'a>(&'a self) -> &'a MiriEvalContext<'mir, 'tcx>;
    fn eval_context_mut<'a>(&'a mut self) -> &'a mut MiriEvalContext<'mir, 'tcx>;
}
impl<'mir, 'tcx> MiriEvalContextExt<'mir, 'tcx> for MiriEvalContext<'mir, 'tcx> {
    #[inline(always)]
    fn eval_context_ref(&self) -> &MiriEvalContext<'mir, 'tcx> {
        self
    }
    #[inline(always)]
    fn eval_context_mut(&mut self) -> &mut MiriEvalContext<'mir, 'tcx> {
        self
    }
}

/// Machine hook implementations.
impl<'mir, 'tcx> Machine<'mir, 'tcx> for Evaluator<'mir, 'tcx> {
    type MemoryKind = MiriMemoryKind;

    type FrameExtra = FrameData<'tcx>;
    type MemoryExtra = MemoryExtra;
    type AllocExtra = AllocExtra;
    type PointerTag = Tag;
    type ExtraFnVal = Dlsym;

    type MemoryMap =
        MonoHashMap<AllocId, (MemoryKind<MiriMemoryKind>, Allocation<Tag, Self::AllocExtra>)>;

    const GLOBAL_KIND: Option<MiriMemoryKind> = Some(MiriMemoryKind::Global);

    const PANIC_ON_ALLOC_FAIL: bool = false;

    #[inline(always)]
    fn enforce_alignment(memory_extra: &MemoryExtra) -> bool {
        memory_extra.check_alignment != AlignmentCheck::None
    }

    #[inline(always)]
    fn force_int_for_alignment_check(memory_extra: &Self::MemoryExtra) -> bool {
        memory_extra.check_alignment == AlignmentCheck::Int
    }

    #[inline(always)]
    fn enforce_validity(ecx: &InterpCx<'mir, 'tcx, Self>) -> bool {
        ecx.machine.validate
    }

    #[inline(always)]
    fn enforce_number_validity(ecx: &InterpCx<'mir, 'tcx, Self>) -> bool {
        ecx.machine.enforce_number_validity
    }

    #[inline(always)]
    fn enforce_abi(ecx: &InterpCx<'mir, 'tcx, Self>) -> bool {
        ecx.machine.enforce_abi
    }

    #[inline(always)]
    fn find_mir_or_eval_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        abi: Abi,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(&PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        unwind: StackPopUnwind,
    ) -> InterpResult<'tcx, Option<(&'mir mir::Body<'tcx>, ty::Instance<'tcx>)>> {
        ecx.find_mir_or_eval_fn(instance, abi, args, ret, unwind)
    }

    #[inline(always)]
    fn call_extra_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        fn_val: Dlsym,
        abi: Abi,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(&PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        _unwind: StackPopUnwind,
    ) -> InterpResult<'tcx> {
        ecx.call_dlsym(fn_val, abi, args, ret)
    }

    #[inline(always)]
    fn call_intrinsic(
        ecx: &mut rustc_const_eval::interpret::InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(&PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        unwind: StackPopUnwind,
    ) -> InterpResult<'tcx> {
        ecx.call_intrinsic(instance, args, ret, unwind)
    }

    #[inline(always)]
    fn assert_panic(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        msg: &mir::AssertMessage<'tcx>,
        unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        ecx.assert_panic(msg, unwind)
    }

    #[inline(always)]
    fn abort(_ecx: &mut InterpCx<'mir, 'tcx, Self>, msg: String) -> InterpResult<'tcx, !> {
        throw_machine_stop!(TerminationInfo::Abort(msg))
    }

    #[inline(always)]
    fn binary_ptr_op(
        ecx: &rustc_const_eval::interpret::InterpCx<'mir, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, Tag>,
        right: &ImmTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, (Scalar<Tag>, bool, ty::Ty<'tcx>)> {
        ecx.binary_ptr_op(bin_op, left, right)
    }

    fn thread_local_static_base_pointer(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        def_id: DefId,
    ) -> InterpResult<'tcx, Pointer<Tag>> {
        ecx.get_or_create_thread_local_alloc(def_id)
    }

    fn extern_static_base_pointer(
        memory: &Memory<'mir, 'tcx, Self>,
        def_id: DefId,
    ) -> InterpResult<'tcx, Pointer<Tag>> {
        let attrs = memory.tcx.get_attrs(def_id);
        let link_name = match memory.tcx.sess.first_attr_value_str_by_name(&attrs, sym::link_name) {
            Some(name) => name,
            None => memory.tcx.item_name(def_id),
        };
        if let Some(&ptr) = memory.extra.extern_statics.get(&link_name) {
            Ok(ptr)
        } else {
            throw_unsup_format!("`extern` static {:?} is not supported by Miri", def_id)
        }
    }

    fn init_allocation_extra<'b>(
        mem: &Memory<'mir, 'tcx, Self>,
        id: AllocId,
        alloc: Cow<'b, Allocation>,
        kind: Option<MemoryKind<Self::MemoryKind>>,
    ) -> Cow<'b, Allocation<Self::PointerTag, Self::AllocExtra>> {
        if Some(id) == mem.extra.tracked_alloc_id {
            register_diagnostic(NonHaltingDiagnostic::CreatedAlloc(id));
        }

        let kind = kind.expect("we set our STATIC_KIND so this cannot be None");
        let alloc = alloc.into_owned();
        let stacks = if let Some(stacked_borrows) = &mem.extra.stacked_borrows {
            Some(Stacks::new_allocation(id, alloc.size(), stacked_borrows, kind))
        } else {
            None
        };
        let race_alloc = if let Some(data_race) = &mem.extra.data_race {
            Some(data_race::AllocExtra::new_allocation(&data_race, alloc.size(), kind))
        } else {
            None
        };
        let alloc: Allocation<Tag, Self::AllocExtra> = alloc.convert_tag_add_extra(
            &mem.tcx,
            AllocExtra { stacked_borrows: stacks, data_race: race_alloc },
            |ptr| Evaluator::tag_alloc_base_pointer(mem, ptr),
        );
        Cow::Owned(alloc)
    }

    fn tag_alloc_base_pointer(
        mem: &Memory<'mir, 'tcx, Self>,
        ptr: Pointer<AllocId>,
    ) -> Pointer<Tag> {
        let absolute_addr = intptrcast::GlobalState::rel_ptr_to_addr(&mem, ptr);
        let sb_tag = if let Some(stacked_borrows) = &mem.extra.stacked_borrows {
            stacked_borrows.borrow_mut().base_tag(ptr.provenance)
        } else {
            SbTag::Untagged
        };
        Pointer::new(Tag { alloc_id: ptr.provenance, sb: sb_tag }, Size::from_bytes(absolute_addr))
    }

    #[inline(always)]
    fn ptr_from_addr(
        mem: &Memory<'mir, 'tcx, Self>,
        addr: u64,
    ) -> Pointer<Option<Self::PointerTag>> {
        intptrcast::GlobalState::ptr_from_addr(addr, mem)
    }

    /// Convert a pointer with provenance into an allocation-offset pair,
    /// or a `None` with an absolute address if that conversion is not possible.
    fn ptr_get_alloc(
        mem: &Memory<'mir, 'tcx, Self>,
        ptr: Pointer<Self::PointerTag>,
    ) -> (AllocId, Size) {
        let rel = intptrcast::GlobalState::abs_ptr_to_rel(mem, ptr);
        (ptr.provenance.alloc_id, rel)
    }

    #[inline(always)]
    fn memory_read(
        memory_extra: &Self::MemoryExtra,
        alloc_extra: &AllocExtra,
        tag: Tag,
        range: AllocRange,
    ) -> InterpResult<'tcx> {
        if let Some(data_race) = &alloc_extra.data_race {
            data_race.read(tag.alloc_id, range, memory_extra.data_race.as_ref().unwrap())?;
        }
        if let Some(stacked_borrows) = &alloc_extra.stacked_borrows {
            stacked_borrows.memory_read(
                tag.alloc_id,
                tag.sb,
                range,
                memory_extra.stacked_borrows.as_ref().unwrap(),
            )
        } else {
            Ok(())
        }
    }

    #[inline(always)]
    fn memory_written(
        memory_extra: &mut Self::MemoryExtra,
        alloc_extra: &mut AllocExtra,
        tag: Tag,
        range: AllocRange,
    ) -> InterpResult<'tcx> {
        if let Some(data_race) = &mut alloc_extra.data_race {
            data_race.write(tag.alloc_id, range, memory_extra.data_race.as_mut().unwrap())?;
        }
        if let Some(stacked_borrows) = &mut alloc_extra.stacked_borrows {
            stacked_borrows.memory_written(
                tag.alloc_id,
                tag.sb,
                range,
                memory_extra.stacked_borrows.as_mut().unwrap(),
            )
        } else {
            Ok(())
        }
    }

    #[inline(always)]
    fn memory_deallocated(
        memory_extra: &mut Self::MemoryExtra,
        alloc_extra: &mut AllocExtra,
        tag: Tag,
        range: AllocRange,
    ) -> InterpResult<'tcx> {
        if Some(tag.alloc_id) == memory_extra.tracked_alloc_id {
            register_diagnostic(NonHaltingDiagnostic::FreedAlloc(tag.alloc_id));
        }
        if let Some(data_race) = &mut alloc_extra.data_race {
            data_race.deallocate(tag.alloc_id, range, memory_extra.data_race.as_mut().unwrap())?;
        }
        if let Some(stacked_borrows) = &mut alloc_extra.stacked_borrows {
            stacked_borrows.memory_deallocated(
                tag.alloc_id,
                tag.sb,
                range,
                memory_extra.stacked_borrows.as_mut().unwrap(),
            )
        } else {
            Ok(())
        }
    }

    #[inline(always)]
    fn retag(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        kind: mir::RetagKind,
        place: &PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        if ecx.memory.extra.stacked_borrows.is_some() { ecx.retag(kind, place) } else { Ok(()) }
    }

    #[inline(always)]
    fn init_frame_extra(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        frame: Frame<'mir, 'tcx, Tag>,
    ) -> InterpResult<'tcx, Frame<'mir, 'tcx, Tag, FrameData<'tcx>>> {
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

        let stacked_borrows = ecx.memory.extra.stacked_borrows.as_ref();
        let call_id = stacked_borrows.map_or(NonZeroU64::new(1).unwrap(), |stacked_borrows| {
            stacked_borrows.borrow_mut().new_call()
        });

        let extra = FrameData { call_id, catch_unwind: None, timing };
        Ok(frame.with_extra(extra))
    }

    fn stack<'a>(
        ecx: &'a InterpCx<'mir, 'tcx, Self>,
    ) -> &'a [Frame<'mir, 'tcx, Self::PointerTag, Self::FrameExtra>] {
        ecx.active_thread_stack()
    }

    fn stack_mut<'a>(
        ecx: &'a mut InterpCx<'mir, 'tcx, Self>,
    ) -> &'a mut Vec<Frame<'mir, 'tcx, Self::PointerTag, Self::FrameExtra>> {
        ecx.active_thread_stack_mut()
    }

    #[inline(always)]
    fn after_stack_push(ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        if ecx.memory.extra.stacked_borrows.is_some() { ecx.retag_return_place() } else { Ok(()) }
    }

    #[inline(always)]
    fn after_stack_pop(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        mut frame: Frame<'mir, 'tcx, Tag, FrameData<'tcx>>,
        unwinding: bool,
    ) -> InterpResult<'tcx, StackPopJump> {
        let timing = frame.extra.timing.take();
        let res = ecx.handle_stack_pop(frame.extra, unwinding);
        if let Some(profiler) = ecx.machine.profiler.as_ref() {
            profiler.finish_recording_interval_event(timing.unwrap());
        }
        res
    }
}
