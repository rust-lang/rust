//! Global machine state as well as implementation of the interpreter engine
//! `Machine` trait.

use std::borrow::Cow;
use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;

use rand::rngs::StdRng;

use rustc_data_structures::fx::FxHashMap;
use rustc::mir;
use rustc::ty::{
    self,
    layout::{LayoutOf, Size},
    Ty,
};
use rustc_ast::attr;
use rustc_span::{source_map::Span, symbol::{sym, Symbol}};

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

    /// If this is Some(), then this is a special "catch unwind" frame (the frame of the closure
    /// called by `__rustc_maybe_catch_panic`). When this frame is popped during unwinding a panic,
    /// we stop unwinding, use the `CatchUnwindData` to
    /// store the panic payload, and continue execution in the parent frame.
    pub catch_panic: Option<CatchUnwindData<'tcx>>,
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
    /// Memory for env vars and args, errno, extern statics and other parts of the machine-managed environment.
    Machine,
    /// Rust statics.
    Static,
}

impl Into<MemoryKind<MiriMemoryKind>> for MiriMemoryKind {
    #[inline(always)]
    fn into(self) -> MemoryKind<MiriMemoryKind> {
        MemoryKind::Machine(self)
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
    pub(crate) extern_statics: FxHashMap<Symbol, AllocId>,

    /// The random number generator used for resolving non-determinism.
    /// Needs to be queried by ptr_to_int, hence needs interior mutability.
    pub(crate) rng: RefCell<StdRng>,
}

impl MemoryExtra {
    pub fn new(rng: StdRng, stacked_borrows: bool, tracked_pointer_tag: Option<PtrId>) -> Self {
        let stacked_borrows = if stacked_borrows {
            Some(Rc::new(RefCell::new(stacked_borrows::GlobalState::new(tracked_pointer_tag))))
        } else {
            None
        };
        MemoryExtra {
            stacked_borrows,
            intptrcast: Default::default(),
            extern_statics: FxHashMap::default(),
            rng: RefCell::new(rng),
        }
    }

    /// Sets up the "extern statics" for this machine.
    pub fn init_extern_statics<'mir, 'tcx>(
        this: &mut MiriEvalContext<'mir, 'tcx>,
    ) -> InterpResult<'tcx> {
        match this.tcx.sess.target.target.target_os.as_str() {
            "linux" => {
                // "__cxa_thread_atexit_impl"
                // This should be all-zero, pointer-sized.
                let layout = this.layout_of(this.tcx.types.usize)?;
                let place = this.allocate(layout, MiriMemoryKind::Machine.into());
                this.write_scalar(Scalar::from_machine_usize(0, &*this.tcx), place.into())?;
                this.memory
                    .extra
                    .extern_statics
                    .insert(Symbol::intern("__cxa_thread_atexit_impl"), place.ptr.assert_ptr().alloc_id)
                    .unwrap_none();
            }
            _ => {} // No "extern statics" supported on this platform
        }
        Ok(())
    }
}

/// The machine itself.
pub struct Evaluator<'tcx> {
    /// Environment variables set by `setenv`.
    /// Miri does not expose env vars from the host to the emulated program.
    pub(crate) env_vars: EnvVars,

    /// Program arguments (`Option` because we can only initialize them after creating the ecx).
    /// These are *pointers* to argc/argv because macOS.
    /// We also need the full command line as one string because of Windows.
    pub(crate) argc: Option<Scalar<Tag>>,
    pub(crate) argv: Option<Scalar<Tag>>,
    pub(crate) cmd_line: Option<Scalar<Tag>>,

    /// Last OS error location in memory. It is a 32-bit integer.
    pub(crate) last_error: Option<MPlaceTy<'tcx, Tag>>,

    /// TLS state.
    pub(crate) tls: TlsData<'tcx>,

    /// If enabled, the `env_vars` field is populated with the host env vars during initialization
    /// and random number generation is delegated to the host.
    pub(crate) communicate: bool,

    /// Whether to enforce the validity invariant.
    pub(crate) validate: bool,

    pub(crate) file_handler: FileHandler,
    pub(crate) dir_handler: DirHandler,

    /// The temporary used for storing the argument of
    /// the call to `miri_start_panic` (the panic payload) when unwinding.
    pub(crate) panic_payload: Option<ImmTy<'tcx, Tag>>,
}

impl<'tcx> Evaluator<'tcx> {
    pub(crate) fn new(communicate: bool, validate: bool) -> Self {
        Evaluator {
            // `env_vars` could be initialized properly here if `Memory` were available before
            // calling this method.
            env_vars: EnvVars::default(),
            argc: None,
            argv: None,
            cmd_line: None,
            last_error: None,
            tls: TlsData::default(),
            communicate,
            validate,
            file_handler: Default::default(),
            dir_handler: Default::default(),
            panic_payload: None,
        }
    }
}

/// A rustc InterpCx for Miri.
pub type MiriEvalContext<'mir, 'tcx> = InterpCx<'mir, 'tcx, Evaluator<'tcx>>;

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
impl<'mir, 'tcx> Machine<'mir, 'tcx> for Evaluator<'tcx> {
    type MemoryKinds = MiriMemoryKind;

    type FrameExtra = FrameData<'tcx>;
    type MemoryExtra = MemoryExtra;
    type AllocExtra = AllocExtra;
    type PointerTag = Tag;
    type ExtraFnVal = Dlsym;

    type MemoryMap =
        MonoHashMap<AllocId, (MemoryKind<MiriMemoryKind>, Allocation<Tag, Self::AllocExtra>)>;

    const STATIC_KIND: Option<MiriMemoryKind> = Some(MiriMemoryKind::Static);

    const CHECK_ALIGN: bool = true;

    #[inline(always)]
    fn enforce_validity(ecx: &InterpCx<'mir, 'tcx, Self>) -> bool {
        ecx.machine.validate
    }

    #[inline(always)]
    fn find_mir_or_eval_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _span: Span,
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
        span: Span,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        ecx.call_intrinsic(span, instance, args, ret, unwind)
    }

    #[inline(always)]
    fn assert_panic(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        span: Span,
        msg: &mir::AssertMessage<'tcx>,
        unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        ecx.assert_panic(span, msg, unwind)
    }

    #[inline(always)]
    fn binary_ptr_op(
        ecx: &rustc_mir::interpret::InterpCx<'mir, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: ImmTy<'tcx, Tag>,
        right: ImmTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, (Scalar<Tag>, bool, Ty<'tcx>)> {
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
        let size = Scalar::from_uint(layout.size.bytes(), ecx.pointer_size());

        // Second argument: `align`.
        let align = Scalar::from_uint(layout.align.abi.bytes(), ecx.pointer_size());

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

    fn canonical_alloc_id(mem: &Memory<'mir, 'tcx, Self>, id: AllocId) -> AllocId {
        let tcx = mem.tcx;
        // Figure out if this is an extern static, and if yes, which one.
        let def_id = match tcx.alloc_map.lock().get(id) {
            Some(GlobalAlloc::Static(def_id)) if tcx.is_foreign_item(def_id) => def_id,
            _ => {
                // No need to canonicalize anything.
                return id;
            }
        };
        let attrs = tcx.get_attrs(def_id);
        let link_name = match attr::first_attr_value_str_by_name(&attrs, sym::link_name) {
            Some(name) => name,
            None => tcx.item_name(def_id),
        };
        // Check if we know this one.
        if let Some(canonical_id) = mem.extra.extern_statics.get(&link_name) {
            trace!("canonical_alloc_id: {:?} ({}) -> {:?}", id, link_name, canonical_id);
            *canonical_id
        } else {
            // Return original id; `Memory::get_static_alloc` will throw an error.
            id
        }
    }

    fn init_allocation_extra<'b>(
        memory_extra: &MemoryExtra,
        id: AllocId,
        alloc: Cow<'b, Allocation>,
        kind: Option<MemoryKind<Self::MemoryKinds>>,
    ) -> (Cow<'b, Allocation<Self::PointerTag, Self::AllocExtra>>, Self::PointerTag) {
        let kind = kind.expect("we set our STATIC_KIND so this cannot be None");
        let alloc = alloc.into_owned();
        let (stacks, base_tag) =
            if let Some(stacked_borrows) = memory_extra.stacked_borrows.as_ref() {
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
                if let Some(stacked_borrows) = stacked_borrows.as_mut() {
                    // Only statics may already contain pointers at this point
                    assert_eq!(kind, MiriMemoryKind::Static.into());
                    stacked_borrows.static_base_ptr(alloc)
                } else {
                    Tag::Untagged
                }
            },
            AllocExtra { stacked_borrows: stacks },
        );
        (Cow::Owned(alloc), base_tag)
    }

    #[inline(always)]
    fn tag_static_base_pointer(memory_extra: &MemoryExtra, id: AllocId) -> Self::PointerTag {
        if let Some(stacked_borrows) = memory_extra.stacked_borrows.as_ref() {
            stacked_borrows.borrow_mut().static_base_ptr(id)
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
        if ecx.memory.extra.stacked_borrows.is_none() {
            // No tracking.
            Ok(())
        } else {
            ecx.retag(kind, place)
        }
    }

    #[inline(always)]
    fn stack_push(ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx, FrameData<'tcx>> {
        let stacked_borrows = ecx.memory.extra.stacked_borrows.as_ref();
        let call_id = stacked_borrows.map_or(NonZeroU64::new(1).unwrap(), |stacked_borrows| {
            stacked_borrows.borrow_mut().new_call()
        });
        Ok(FrameData { call_id, catch_panic: None })
    }

    #[inline(always)]
    fn stack_pop(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        extra: FrameData<'tcx>,
        unwinding: bool,
    ) -> InterpResult<'tcx, StackPopInfo> {
        ecx.handle_stack_pop(extra, unwinding)
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
        if let Some(ref stacked_borrows) = alloc.extra.stacked_borrows {
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
        if let Some(ref mut stacked_borrows) = alloc.extra.stacked_borrows {
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
        if let Some(ref mut stacked_borrows) = alloc.extra.stacked_borrows {
            stacked_borrows.memory_deallocated(ptr, size)
        } else {
            Ok(())
        }
    }
}

impl MayLeak for MiriMemoryKind {
    #[inline(always)]
    fn may_leak(self) -> bool {
        use self::MiriMemoryKind::*;
        match self {
            Rust | C | WinHeap => false,
            Machine | Static => true,
        }
    }
}
