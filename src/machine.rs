//! Global machine state as well as implementation of the interpreter engine
//! `Machine` trait.

use std::borrow::Cow;
use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::time::Instant;
use std::fmt;

use log::trace;
use rand::rngs::StdRng;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::{
    mir,
    ty::{
        self,
        layout::{LayoutCx, LayoutError, TyAndLayout},
        TyCtxt,
    },
};
use rustc_span::symbol::{sym, Symbol};
use rustc_span::def_id::DefId;
use rustc_target::abi::{LayoutOf, Size};

use crate::*;

// Some global facts about the emulated machine.
pub const PAGE_SIZE: u64 = 4 * 1024; // FIXME: adjust to target architecture
pub const STACK_ADDR: u64 = 32 * PAGE_SIZE; // not really about the "stack", but where we start assigning integer addresses to allocations
pub const STACK_SIZE: u64 = 16 * PAGE_SIZE; // whatever
pub const NUM_CPUS: u64 = 1;

/// Extra data stored with each stack frame
#[derive(Debug)]
pub struct FrameData<'tcx> {
    /// Extra data for Stacked Borrows.
    pub call_id: stacked_borrows::CallId,

    /// If this is Some(), then this is a special "catch unwind" frame (the frame of `try_fn`
    /// called by `try`). When this frame is popped during unwinding a panic,
    /// we stop unwinding, use the `CatchUnwindData` to handle catching.
    pub catch_unwind: Option<CatchUnwindData<'tcx>>,
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
    /// Memory for env vars. Separate from `Machine` because we clean it up and leak-check it.
    Env,
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
            Rust | C | WinHeap | Env => false,
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
            Env => write!(f, "environment variable"),
            Global => write!(f, "global (static or const)"),
            ExternStatic => write!(f, "extern static"),
            Tls =>  write!(f, "thread-local static"),
        }
    }
}

/// Extra per-allocation data
#[derive(Debug, Clone)]
pub struct AllocExtra {
    /// Stacked Borrows state is only added if it is enabled.
    pub stacked_borrows: Option<stacked_borrows::AllocExtra>,
}

/// Extra global memory data
#[derive(Clone, Debug)]
pub struct MemoryExtra {
    pub stacked_borrows: Option<stacked_borrows::MemoryExtra>,
    pub intptrcast: intptrcast::MemoryExtra,

    /// Mapping extern static names to their canonical allocation.
    extern_statics: FxHashMap<Symbol, AllocId>,

    /// The random number generator used for resolving non-determinism.
    /// Needs to be queried by ptr_to_int, hence needs interior mutability.
    pub(crate) rng: RefCell<StdRng>,

    /// An allocation ID to report when it is being allocated
    /// (helps for debugging memory leaks and use after free bugs).
    tracked_alloc_id: Option<AllocId>,

    /// Controls whether alignment of memory accesses is being checked.
    pub(crate) check_alignment: AlignmentCheck,
}

impl MemoryExtra {
    pub fn new(
        rng: StdRng,
        stacked_borrows: bool,
        tracked_pointer_tag: Option<PtrId>,
        tracked_call_id: Option<CallId>,
        tracked_alloc_id: Option<AllocId>,
        check_alignment: AlignmentCheck,
    ) -> Self {
        let stacked_borrows = if stacked_borrows {
            Some(Rc::new(RefCell::new(stacked_borrows::GlobalState::new(tracked_pointer_tag, tracked_call_id))))
        } else {
            None
        };
        MemoryExtra {
            stacked_borrows,
            intptrcast: Default::default(),
            extern_statics: FxHashMap::default(),
            rng: RefCell::new(rng),
            tracked_alloc_id,
            check_alignment,
        }
    }

    fn add_extern_static<'tcx, 'mir>(
        this: &mut MiriEvalContext<'mir, 'tcx>,
        name: &str,
        ptr: Scalar<Tag>,
    ) {
        let ptr = ptr.assert_ptr();
        assert_eq!(ptr.offset, Size::ZERO);
        this.memory
            .extra
            .extern_statics
            .insert(Symbol::intern(name), ptr.alloc_id)
            .unwrap_none();
    }

    /// Sets up the "extern statics" for this machine.
    pub fn init_extern_statics<'tcx, 'mir>(
        this: &mut MiriEvalContext<'mir, 'tcx>,
    ) -> InterpResult<'tcx> {
        match this.tcx.sess.target.target.target_os.as_str() {
            "linux" => {
                // "__cxa_thread_atexit_impl"
                // This should be all-zero, pointer-sized.
                let layout = this.machine.layouts.usize;
                let place = this.allocate(layout, MiriMemoryKind::ExternStatic.into());
                this.write_scalar(Scalar::from_machine_usize(0, this), place.into())?;
                Self::add_extern_static(this, "__cxa_thread_atexit_impl", place.ptr);
                // "environ"
                Self::add_extern_static(this, "environ", this.machine.env_vars.environ.unwrap().ptr);
            }
            "windows" => {
                // "_tls_used"
                // This is some obscure hack that is part of the Windows TLS story. It's a `u8`.
                let layout = this.machine.layouts.u8;
                let place = this.allocate(layout, MiriMemoryKind::ExternStatic.into());
                this.write_scalar(Scalar::from_u8(0), place.into())?;
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
    pub(crate) argc: Option<Scalar<Tag>>,
    pub(crate) argv: Option<Scalar<Tag>>,
    pub(crate) cmd_line: Option<Scalar<Tag>>,

    /// TLS state.
    pub(crate) tls: TlsData<'tcx>,

    /// If enabled, the `env_vars` field is populated with the host env vars during initialization
    /// and random number generation is delegated to the host.
    pub(crate) communicate: bool,

    /// Whether to enforce the validity invariant.
    pub(crate) validate: bool,

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
}

impl<'mir, 'tcx> Evaluator<'mir, 'tcx> {
    pub(crate) fn new(
        communicate: bool,
        validate: bool,
        layout_cx: LayoutCx<'tcx, TyCtxt<'tcx>>,
    ) -> Self {
        let layouts = PrimitiveLayouts::new(layout_cx)
            .expect("Couldn't get layouts of primitive types");
        Evaluator {
            // `env_vars` could be initialized properly here if `Memory` were available before
            // calling this method.
            env_vars: EnvVars::default(),
            argc: None,
            argv: None,
            cmd_line: None,
            tls: TlsData::default(),
            communicate,
            validate,
            file_handler: Default::default(),
            dir_handler: Default::default(),
            time_anchor: Instant::now(),
            layouts,
            threads: ThreadManager::default(),
            static_roots: Vec::new(),
        }
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
    fn find_mir_or_eval_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, Option<&'mir mir::Body<'tcx>>> {
        ecx.find_mir_or_eval_fn(instance, args, ret, unwind)
    }

    #[inline(always)]
    fn call_extra_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        fn_val: Dlsym,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        _unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        ecx.call_dlsym(fn_val, args, ret)
    }

    #[inline(always)]
    fn call_intrinsic(
        ecx: &mut rustc_mir::interpret::InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        unwind: Option<mir::BasicBlock>,
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
    fn abort(_ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx, !> {
        throw_machine_stop!(TerminationInfo::Abort(None))
    }

    #[inline(always)]
    fn binary_ptr_op(
        ecx: &rustc_mir::interpret::InterpCx<'mir, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: ImmTy<'tcx, Tag>,
        right: ImmTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, (Scalar<Tag>, bool, ty::Ty<'tcx>)> {
        ecx.binary_ptr_op(bin_op, left, right)
    }

    fn box_alloc(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        dest: PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        trace!("box_alloc for {:?}", dest.layout.ty);
        let layout = ecx.layout_of(dest.layout.ty.builtin_deref(false).unwrap().ty)?;
        // First argument: `size`.
        // (`0` is allowed here -- this is expected to be handled by the lang item).
        let size = Scalar::from_machine_usize(layout.size.bytes(), ecx);

        // Second argument: `align`.
        let align = Scalar::from_machine_usize(layout.align.abi.bytes(), ecx);

        // Call the `exchange_malloc` lang item.
        let malloc = ecx.tcx.lang_items().exchange_malloc_fn().unwrap();
        let malloc = ty::Instance::mono(ecx.tcx.tcx, malloc);
        ecx.call_function(
            malloc,
            &[size.into(), align.into()],
            Some(dest),
            // Don't do anything when we are done. The `statement()` function will increment
            // the old stack frame's stmt counter to the next statement, which means that when
            // `exchange_malloc` returns, we go on evaluating exactly where we want to be.
            StackPopCleanup::None { cleanup: true },
        )?;
        Ok(())
    }

    fn thread_local_static_alloc_id(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        def_id: DefId,
    ) -> InterpResult<'tcx, AllocId> {
        ecx.get_or_create_thread_local_alloc_id(def_id)
    }

    fn extern_static_alloc_id(
        memory: &Memory<'mir, 'tcx, Self>,
        def_id: DefId,
    ) -> InterpResult<'tcx, AllocId> {
        let attrs = memory.tcx.get_attrs(def_id);
        let link_name = match memory.tcx.sess.first_attr_value_str_by_name(&attrs, sym::link_name) {
            Some(name) => name,
            None => memory.tcx.item_name(def_id),
        };
        if let Some(&id) = memory.extra.extern_statics.get(&link_name) {
            Ok(id)
        } else {
            throw_unsup_format!("`extern` static {:?} is not supported by Miri", def_id)
        }
    }

    fn init_allocation_extra<'b>(
        memory_extra: &MemoryExtra,
        id: AllocId,
        alloc: Cow<'b, Allocation>,
        kind: Option<MemoryKind<Self::MemoryKind>>,
    ) -> (Cow<'b, Allocation<Self::PointerTag, Self::AllocExtra>>, Self::PointerTag) {
        if Some(id) == memory_extra.tracked_alloc_id {
            register_diagnostic(NonHaltingDiagnostic::CreatedAlloc(id));
        }

        let kind = kind.expect("we set our STATIC_KIND so this cannot be None");
        let alloc = alloc.into_owned();
        let (stacks, base_tag) =
            if let Some(stacked_borrows) = &memory_extra.stacked_borrows {
                let (stacks, base_tag) =
                    Stacks::new_allocation(id, alloc.size, Rc::clone(stacked_borrows), kind);
                (Some(stacks), base_tag)
            } else {
                // No stacks, no tag.
                (None, Tag::Untagged)
            };
        let mut stacked_borrows = memory_extra.stacked_borrows.as_ref().map(|sb| sb.borrow_mut());
        let alloc: Allocation<Tag, Self::AllocExtra> = alloc.with_tags_and_extra(
            |alloc| {
                if let Some(stacked_borrows) = &mut stacked_borrows {
                    // Only globals may already contain pointers at this point
                    assert_eq!(kind, MiriMemoryKind::Global.into());
                    stacked_borrows.global_base_ptr(alloc)
                } else {
                    Tag::Untagged
                }
            },
            AllocExtra { stacked_borrows: stacks },
        );
        (Cow::Owned(alloc), base_tag)
    }

    #[inline(always)]
    fn before_deallocation(
        memory_extra: &mut Self::MemoryExtra,
        id: AllocId,
    ) -> InterpResult<'tcx> {
        if Some(id) == memory_extra.tracked_alloc_id {
            register_diagnostic(NonHaltingDiagnostic::FreedAlloc(id));
        }

        Ok(())
    }

    #[inline(always)]
    fn tag_global_base_pointer(memory_extra: &MemoryExtra, id: AllocId) -> Self::PointerTag {
        if let Some(stacked_borrows) = &memory_extra.stacked_borrows {
            stacked_borrows.borrow_mut().global_base_ptr(id)
        } else {
            Tag::Untagged
        }
    }

    #[inline(always)]
    fn retag(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        kind: mir::RetagKind,
        place: PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        if ecx.memory.extra.stacked_borrows.is_some() {
            ecx.retag(kind, place)
        } else {
            Ok(())
        }
    }

    #[inline(always)]
    fn init_frame_extra(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        frame: Frame<'mir, 'tcx, Tag>,
    ) -> InterpResult<'tcx, Frame<'mir, 'tcx, Tag, FrameData<'tcx>>> {
        let stacked_borrows = ecx.memory.extra.stacked_borrows.as_ref();
        let call_id = stacked_borrows.map_or(NonZeroU64::new(1).unwrap(), |stacked_borrows| {
            stacked_borrows.borrow_mut().new_call()
        });
        let extra = FrameData { call_id, catch_unwind: None };
        Ok(frame.with_extra(extra))
    }

    fn stack<'a>(
        ecx: &'a InterpCx<'mir, 'tcx, Self>
    ) -> &'a [Frame<'mir, 'tcx, Self::PointerTag, Self::FrameExtra>] {
        ecx.active_thread_stack()
    }

    fn stack_mut<'a>(
        ecx: &'a mut InterpCx<'mir, 'tcx, Self>
    ) -> &'a mut Vec<Frame<'mir, 'tcx, Self::PointerTag, Self::FrameExtra>> {
        ecx.active_thread_stack_mut()
    }

    #[inline(always)]
    fn after_stack_push(ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        if ecx.memory.extra.stacked_borrows.is_some() {
            ecx.retag_return_place()
        } else {
            Ok(())
        }
    }

    #[inline(always)]
    fn after_stack_pop(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        frame: Frame<'mir, 'tcx, Tag, FrameData<'tcx>>,
        unwinding: bool,
    ) -> InterpResult<'tcx, StackPopJump> {
        ecx.handle_stack_pop(frame.extra, unwinding)
    }

    #[inline(always)]
    fn int_to_ptr(
        memory: &Memory<'mir, 'tcx, Self>,
        int: u64,
    ) -> InterpResult<'tcx, Pointer<Self::PointerTag>> {
        intptrcast::GlobalState::int_to_ptr(int, memory)
    }

    #[inline(always)]
    fn ptr_to_int(
        memory: &Memory<'mir, 'tcx, Self>,
        ptr: Pointer<Self::PointerTag>,
    ) -> InterpResult<'tcx, u64> {
        intptrcast::GlobalState::ptr_to_int(ptr, memory)
    }
}

impl AllocationExtra<Tag> for AllocExtra {
    #[inline(always)]
    fn memory_read<'tcx>(
        alloc: &Allocation<Tag, AllocExtra>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx> {
        if let Some(stacked_borrows) = &alloc.extra.stacked_borrows {
            stacked_borrows.memory_read(ptr, size)
        } else {
            Ok(())
        }
    }

    #[inline(always)]
    fn memory_written<'tcx>(
        alloc: &mut Allocation<Tag, AllocExtra>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx> {
        if let Some(stacked_borrows) = &mut alloc.extra.stacked_borrows {
            stacked_borrows.memory_written(ptr, size)
        } else {
            Ok(())
        }
    }

    #[inline(always)]
    fn memory_deallocated<'tcx>(
        alloc: &mut Allocation<Tag, AllocExtra>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx> {
        if let Some(stacked_borrows) = &mut alloc.extra.stacked_borrows {
            stacked_borrows.memory_deallocated(ptr, size)
        } else {
            Ok(())
        }
    }
}
