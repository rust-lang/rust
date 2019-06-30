use std::rc::Rc;
use std::borrow::Cow;
use std::collections::HashMap;
use std::cell::RefCell;

use rand::rngs::StdRng;

use syntax::attr;
use syntax::symbol::sym;
use rustc::hir::def_id::DefId;
use rustc::ty::{self, layout::{Size, LayoutOf}, query::TyCtxtAt};
use rustc::mir;

use crate::*;

// Some global facts about the emulated machine.
pub const PAGE_SIZE: u64 = 4*1024; // FIXME: adjust to target architecture
pub const STACK_ADDR: u64 = 16*PAGE_SIZE; // not really about the "stack", but where we start assigning integer addresses to allocations
pub const NUM_CPUS: u64 = 1;

/// Extra memory kinds
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MiriMemoryKind {
    /// `__rust_alloc` memory.
    Rust,
    /// `malloc` memory.
    C,
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
    pub stacked_borrows: stacked_borrows::AllocExtra,
}

/// Extra global memory data
#[derive(Default, Clone, Debug)]
pub struct MemoryExtra {
    pub stacked_borrows: stacked_borrows::MemoryExtra,
    pub intptrcast: intptrcast::MemoryExtra,
    /// The random number generator to use if Miri is running in non-deterministic mode and to
    /// enable intptrcast
    pub(crate) rng: Option<RefCell<StdRng>>
}

impl MemoryExtra {
    pub fn with_rng(rng: Option<StdRng>) -> Self {
        MemoryExtra {
            stacked_borrows: Default::default(),
            intptrcast: Default::default(),
            rng: rng.map(RefCell::new),
        }
    }
}

/// The machine itself.
pub struct Evaluator<'tcx> {
    /// Environment variables set by `setenv`.
    /// Miri does not expose env vars from the host to the emulated program.
    pub(crate) env_vars: HashMap<Vec<u8>, Pointer<Tag>>,

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

    /// Whether to enforce the validity invariant.
    pub(crate) validate: bool,
}

impl<'tcx> Evaluator<'tcx> {
    pub(crate) fn new(validate: bool) -> Self {
        Evaluator {
            env_vars: HashMap::default(),
            argc: None,
            argv: None,
            cmd_line: None,
            last_error: 0,
            tls: TlsData::default(),
            validate,
        }
    }
}

/// A rustc InterpretCx for Miri.
pub type MiriEvalContext<'mir, 'tcx> = InterpretCx<'mir, 'tcx, Evaluator<'tcx>>;

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

    type MemoryMap = MonoHashMap<AllocId, (MemoryKind<MiriMemoryKind>, Allocation<Tag, Self::AllocExtra>)>;

    const STATIC_KIND: Option<MiriMemoryKind> = Some(MiriMemoryKind::Static);

    #[inline(always)]
    fn enforce_validity(ecx: &InterpretCx<'mir, 'tcx, Self>) -> bool {
        ecx.machine.validate
    }

    /// Returns `Ok()` when the function was handled; fail otherwise.
    #[inline(always)]
    fn find_fn(
        ecx: &mut InterpretCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        dest: Option<PlaceTy<'tcx, Tag>>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, Option<&'mir mir::Body<'tcx>>> {
        ecx.find_fn(instance, args, dest, ret)
    }

    #[inline(always)]
    fn call_intrinsic(
        ecx: &mut rustc_mir::interpret::InterpretCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        dest: PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        ecx.call_intrinsic(instance, args, dest)
    }

    #[inline(always)]
    fn ptr_op(
        ecx: &rustc_mir::interpret::InterpretCx<'mir, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: ImmTy<'tcx, Tag>,
        right: ImmTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, (Scalar<Tag>, bool)> {
        ecx.ptr_op(bin_op, left, right)
    }

    fn box_alloc(
        ecx: &mut InterpretCx<'mir, 'tcx, Self>,
        dest: PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        trace!("box_alloc for {:?}", dest.layout.ty);
        // Call the `exchange_malloc` lang item.
        let malloc = ecx.tcx.lang_items().exchange_malloc_fn().unwrap();
        let malloc = ty::Instance::mono(ecx.tcx.tcx, malloc);
        let malloc_mir = ecx.load_mir(malloc.def)?;
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
        let arg = ecx.eval_place(&mir::Place::Base(mir::PlaceBase::Local(args.next().unwrap())))?;
        let size = layout.size.bytes();
        ecx.write_scalar(Scalar::from_uint(size, arg.layout.size), arg)?;

        // Second argument: `align`.
        let arg = ecx.eval_place(&mir::Place::Base(mir::PlaceBase::Local(args.next().unwrap())))?;
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
        def_id: DefId,
        tcx: TyCtxtAt<'tcx>,
    ) -> InterpResult<'tcx, Cow<'tcx, Allocation>> {
        let attrs = tcx.get_attrs(def_id);
        let link_name = match attr::first_attr_value_str_by_name(&attrs, sym::link_name) {
            Some(name) => name.as_str(),
            None => tcx.item_name(def_id).as_str(),
        };

        let alloc = match link_name.get() {
            "__cxa_thread_atexit_impl" => {
                // This should be all-zero, pointer-sized.
                let size = tcx.data_layout.pointer_size;
                let data = vec![0; size.bytes() as usize];
                Allocation::from_bytes(&data, tcx.data_layout.pointer_align.abi)
            }
            _ => return err!(Unimplemented(
                    format!("can't access foreign static: {}", link_name),
                )),
        };
        Ok(Cow::Owned(alloc))
    }

    #[inline(always)]
    fn before_terminator(_ecx: &mut InterpretCx<'mir, 'tcx, Self>) -> InterpResult<'tcx>
    {
        // We are not interested in detecting loops.
        Ok(())
    }

    fn tag_allocation<'b>(
        id: AllocId,
        alloc: Cow<'b, Allocation>,
        kind: Option<MemoryKind<Self::MemoryKinds>>,
        memory: &Memory<'mir, 'tcx, Self>,
    ) -> (Cow<'b, Allocation<Self::PointerTag, Self::AllocExtra>>, Self::PointerTag) {
        let kind = kind.expect("we set our STATIC_KIND so this cannot be None");
        let alloc = alloc.into_owned();
        let (stacks, base_tag) = Stacks::new_allocation(
            id,
            Size::from_bytes(alloc.bytes.len() as u64),
            Rc::clone(&memory.extra.stacked_borrows),
            kind,
        );
        if kind != MiriMemoryKind::Static.into() {
            assert!(alloc.relocations.is_empty(), "Only statics can come initialized with inner pointers");
            // Now we can rely on the inner pointers being static, too.
        }
        let mut memory_extra = memory.extra.stacked_borrows.borrow_mut();
        let alloc: Allocation<Tag, Self::AllocExtra> = Allocation {
            bytes: alloc.bytes,
            relocations: Relocations::from_presorted(
                alloc.relocations.iter()
                    // The allocations in the relocations (pointers stored *inside* this allocation)
                    // all get the base pointer tag.
                    .map(|&(offset, ((), alloc))| (offset, (memory_extra.static_base_ptr(alloc), alloc)))
                    .collect()
            ),
            undef_mask: alloc.undef_mask,
            align: alloc.align,
            mutability: alloc.mutability,
            extra: AllocExtra {
                stacked_borrows: stacks,
            },
        };
        (Cow::Owned(alloc), base_tag)
    }

    #[inline(always)]
    fn tag_static_base_pointer(
        id: AllocId,
        memory: &Memory<'mir, 'tcx, Self>,
    ) -> Self::PointerTag {
        memory.extra.stacked_borrows.borrow_mut().static_base_ptr(id)
    }

    #[inline(always)]
    fn retag(
        ecx: &mut InterpretCx<'mir, 'tcx, Self>,
        kind: mir::RetagKind,
        place: PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        if !ecx.tcx.sess.opts.debugging_opts.mir_emit_retag || !Self::enforce_validity(ecx) {
            // No tracking, or no retagging. The latter is possible because a dependency of ours
            // might be called with different flags than we are, so there are `Retag`
            // statements but we do not want to execute them.
            // Also, honor the whitelist in `enforce_validity` because otherwise we might retag
            // uninitialized data.
             Ok(())
        } else {
            ecx.retag(kind, place)
        }
    }

    #[inline(always)]
    fn stack_push(
        ecx: &mut InterpretCx<'mir, 'tcx, Self>,
    ) -> InterpResult<'tcx, stacked_borrows::CallId> {
        Ok(ecx.memory().extra.stacked_borrows.borrow_mut().new_call())
    }

    #[inline(always)]
    fn stack_pop(
        ecx: &mut InterpretCx<'mir, 'tcx, Self>,
        extra: stacked_borrows::CallId,
    ) -> InterpResult<'tcx> {
        Ok(ecx.memory().extra.stacked_borrows.borrow_mut().end_call(extra))
    }

    fn int_to_ptr(
        int: u64,
        memory: &Memory<'mir, 'tcx, Self>,
    ) -> InterpResult<'tcx, Pointer<Self::PointerTag>> {
        if int == 0 {
            err!(InvalidNullPointerUsage)
        } else if memory.extra.rng.is_none() {
            err!(ReadBytesAsPointer)
        } else {
           intptrcast::GlobalState::int_to_ptr(int, memory)
        }
    }

    fn ptr_to_int(
        ptr: Pointer<Self::PointerTag>,
        memory: &Memory<'mir, 'tcx, Self>,
    ) -> InterpResult<'tcx, u64> {
        if memory.extra.rng.is_none() {
            err!(ReadPointerAsBytes)
        } else {
            intptrcast::GlobalState::ptr_to_int(ptr, memory)
        }
    }
}

impl AllocationExtra<Tag> for AllocExtra {
    #[inline(always)]
    fn memory_read<'tcx>(
        alloc: &Allocation<Tag, AllocExtra>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx> {
        alloc.extra.stacked_borrows.memory_read(ptr, size)
    }

    #[inline(always)]
    fn memory_written<'tcx>(
        alloc: &mut Allocation<Tag, AllocExtra>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx> {
        alloc.extra.stacked_borrows.memory_written(ptr, size)
    }

    #[inline(always)]
    fn memory_deallocated<'tcx>(
        alloc: &mut Allocation<Tag, AllocExtra>,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx> {
        alloc.extra.stacked_borrows.memory_deallocated(ptr, size)
    }
}

impl MayLeak for MiriMemoryKind {
    #[inline(always)]
    fn may_leak(self) -> bool {
        use self::MiriMemoryKind::*;
        match self {
            Rust | C => false,
            Env | Static => true,
        }
    }
}
