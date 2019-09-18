//! Global machine state as well as implementation of the interpreter engine
//! `Machine` trait.

use std::rc::Rc;
use std::borrow::Cow;
use std::cell::RefCell;

use rand::rngs::StdRng;

use syntax::attr;
use syntax::symbol::sym;
use rustc::hir::def_id::DefId;
use rustc::ty::{self, Ty, TyCtxt, layout::{Size, LayoutOf}};
use rustc::mir;

use crate::*;

// Some global facts about the emulated machine.
pub const PAGE_SIZE: u64 = 4*1024; // FIXME: adjust to target architecture
pub const STACK_ADDR: u64 = 32*PAGE_SIZE; // not really about the "stack", but where we start assigning integer addresses to allocations
pub const STACK_SIZE: u64 = 16*PAGE_SIZE; // whatever
pub const NUM_CPUS: u64 = 1;

/// Extra memory kinds
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MiriMemoryKind {
    /// `__rust_alloc` memory.
    Rust,
    /// `malloc` memory.
    C,
    /// Windows `HeapAlloc` memory.
    WinHeap,
    /// Part of env var emulation.
    Env,
    /// Statics.
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
    /// Stacked Borrows state is only added if validation is enabled.
    pub stacked_borrows: Option<stacked_borrows::AllocExtra>,
}

/// Extra global memory data
#[derive(Clone, Debug)]
pub struct MemoryExtra {
    pub stacked_borrows: stacked_borrows::MemoryExtra,
    pub intptrcast: intptrcast::MemoryExtra,

    /// The random number generator used for resolving non-determinism.
    pub(crate) rng: RefCell<StdRng>,

    /// Whether to enforce the validity invariant.
    pub(crate) validate: bool,
}

impl MemoryExtra {
    pub fn new(rng: StdRng, validate: bool) -> Self {
        MemoryExtra {
            stacked_borrows: Default::default(),
            intptrcast: Default::default(),
            rng: RefCell::new(rng),
            validate,
        }
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
    pub(crate) argc: Option<Pointer<Tag>>,
    pub(crate) argv: Option<Pointer<Tag>>,
    pub(crate) cmd_line: Option<Pointer<Tag>>,

    /// Last OS error.
    pub(crate) last_error: u32,

    /// TLS state.
    pub(crate) tls: TlsData<'tcx>,

    /// If enabled, the `env_vars` field is populated with the host env vars during initialization
    /// and random number generation is delegated to the host.
    pub(crate) communicate: bool,
}

impl<'tcx> Evaluator<'tcx> {
    pub(crate) fn new(communicate: bool) -> Self {
        Evaluator {
            // `env_vars` could be initialized properly here if `Memory` were available before
            // calling this method.
            env_vars: EnvVars::default(),
            argc: None,
            argv: None,
            cmd_line: None,
            last_error: 0,
            tls: TlsData::default(),
            communicate,
        }
    }
}

/// A rustc InterpCx for Miri.
pub type MiriEvalContext<'mir, 'tcx> = InterpCx<'mir, 'tcx, Evaluator<'tcx>>;

/// A little trait that's useful to be inherited by extension traits.
pub trait MiriEvalContextExt<'mir, 'tcx> {
    fn eval_context_ref(&self) -> &MiriEvalContext<'mir, 'tcx>;
    fn eval_context_mut(&mut self) -> &mut MiriEvalContext<'mir, 'tcx>;
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

    type FrameExtra = stacked_borrows::CallId;
    type MemoryExtra = MemoryExtra;
    type AllocExtra = AllocExtra;
    type PointerTag = Tag;
    type ExtraFnVal = Dlsym;

    type MemoryMap = MonoHashMap<AllocId, (MemoryKind<MiriMemoryKind>, Allocation<Tag, Self::AllocExtra>)>;

    const STATIC_KIND: Option<MiriMemoryKind> = Some(MiriMemoryKind::Static);

    const CHECK_ALIGN: bool = true;

    #[inline(always)]
    fn enforce_validity(ecx: &InterpCx<'mir, 'tcx, Self>) -> bool {
        ecx.memory().extra.validate
    }

    #[inline(always)]
    fn find_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        dest: Option<PlaceTy<'tcx, Tag>>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, Option<&'mir mir::Body<'tcx>>> {
        ecx.find_fn(instance, args, dest, ret)
    }

    #[inline(always)]
    fn call_extra_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        fn_val: Dlsym,
        args: &[OpTy<'tcx, Tag>],
        dest: Option<PlaceTy<'tcx, Tag>>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        ecx.call_dlsym(fn_val, args, dest, ret)
    }

    #[inline(always)]
    fn call_intrinsic(
        ecx: &mut rustc_mir::interpret::InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        dest: PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        ecx.call_intrinsic(instance, args, dest)
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
        // Call the `exchange_malloc` lang item.
        let malloc = ecx.tcx.lang_items().exchange_malloc_fn().unwrap();
        let malloc = ty::Instance::mono(ecx.tcx.tcx, malloc);
        let malloc_mir = ecx.load_mir(malloc.def, None)?;
        ecx.push_stack_frame(
            malloc,
            malloc_mir.span,
            malloc_mir,
            Some(dest),
            // Don't do anything when we are done. The `statement()` function will increment
            // the old stack frame's stmt counter to the next statement, which means that when
            // `exchange_malloc` returns, we go on evaluating exactly where we want to be.
            StackPopCleanup::None { cleanup: true },
        )?;

        let mut args = ecx.frame().body.args_iter();
        let layout = ecx.layout_of(dest.layout.ty.builtin_deref(false).unwrap().ty)?;

        // First argument: `size`.
        // (`0` is allowed here -- this is expected to be handled by the lang item).
        let arg = ecx.local_place(args.next().unwrap())?;
        let size = layout.size.bytes();
        ecx.write_scalar(Scalar::from_uint(size, arg.layout.size), arg)?;

        // Second argument: `align`.
        let arg = ecx.local_place(args.next().unwrap())?;
        let align = layout.align.abi.bytes();
        ecx.write_scalar(Scalar::from_uint(align, arg.layout.size), arg)?;

        // No more arguments.
        assert!(
            args.next().is_none(),
            "`exchange_malloc` lang item has more arguments than expected"
        );
        Ok(())
    }

    fn find_foreign_static(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
    ) -> InterpResult<'tcx, Cow<'tcx, Allocation>> {
        let attrs = tcx.get_attrs(def_id);
        let link_name = match attr::first_attr_value_str_by_name(&attrs, sym::link_name) {
            Some(name) => name.as_str(),
            None => tcx.item_name(def_id).as_str(),
        };

        let alloc = match &*link_name {
            "__cxa_thread_atexit_impl" => {
                // This should be all-zero, pointer-sized.
                let size = tcx.data_layout.pointer_size;
                let data = vec![0; size.bytes() as usize];
                Allocation::from_bytes(&data, tcx.data_layout.pointer_align.abi)
            }
            _ => throw_unsup_format!("can't access foreign static: {}", link_name),
        };
        Ok(Cow::Owned(alloc))
    }

    #[inline(always)]
    fn before_terminator(_ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx>
    {
        // We are not interested in detecting loops.
        Ok(())
    }

    fn tag_allocation<'b>(
        memory_extra: &MemoryExtra,
        id: AllocId,
        alloc: Cow<'b, Allocation>,
        kind: Option<MemoryKind<Self::MemoryKinds>>,
    ) -> (Cow<'b, Allocation<Self::PointerTag, Self::AllocExtra>>, Self::PointerTag) {
        let kind = kind.expect("we set our STATIC_KIND so this cannot be None");
        let alloc = alloc.into_owned();
        let (stacks, base_tag) = if !memory_extra.validate {
            (None, Tag::Untagged)
        } else {
            let (stacks, base_tag) = Stacks::new_allocation(
                id,
                alloc.size,
                Rc::clone(&memory_extra.stacked_borrows),
                kind,
            );
            (Some(stacks), base_tag)
        };
        let mut stacked_borrows = memory_extra.stacked_borrows.borrow_mut();
        let alloc: Allocation<Tag, Self::AllocExtra> = alloc.with_tags_and_extra(
            |alloc| if !memory_extra.validate {
                Tag::Untagged
            } else {
                // Only statics may already contain pointers at this point
                assert_eq!(kind, MiriMemoryKind::Static.into());
                stacked_borrows.static_base_ptr(alloc)
            },
            AllocExtra {
                stacked_borrows: stacks,
            },
        );
        (Cow::Owned(alloc), base_tag)
    }

    #[inline(always)]
    fn tag_static_base_pointer(
        memory_extra: &MemoryExtra,
        id: AllocId,
    ) -> Self::PointerTag {
        if !memory_extra.validate {
            Tag::Untagged
        } else {
            memory_extra.stacked_borrows.borrow_mut().static_base_ptr(id)
        }
    }

    #[inline(always)]
    fn retag(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        kind: mir::RetagKind,
        place: PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        if !Self::enforce_validity(ecx) {
            // No tracking.
             Ok(())
        } else {
            ecx.retag(kind, place)
        }
    }

    #[inline(always)]
    fn stack_push(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
    ) -> InterpResult<'tcx, stacked_borrows::CallId> {
        Ok(ecx.memory().extra.stacked_borrows.borrow_mut().new_call())
    }

    #[inline(always)]
    fn stack_pop(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        extra: stacked_borrows::CallId,
    ) -> InterpResult<'tcx> {
        Ok(ecx.memory().extra.stacked_borrows.borrow_mut().end_call(extra))
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
            Env | Static => true,
        }
    }
}
