use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use rustc::hir::def_id::DefId;
use rustc::hir::map::definitions::DefPathData;
use rustc::middle::const_val::ConstVal;
use rustc::middle::region::CodeExtent;
use rustc::mir;
use rustc::traits::Reveal;
use rustc::ty::layout::{self, Layout, Size};
use rustc::ty::subst::{Subst, Substs, Kind};
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable, Binder};
use rustc::traits;
use rustc_data_structures::indexed_vec::Idx;
use syntax::codemap::{self, DUMMY_SP, Span};
use syntax::ast::{self, Mutability};
use syntax::abi::Abi;

use error::{EvalError, EvalResult};
use lvalue::{Global, GlobalId, Lvalue, LvalueExtra};
use memory::{Memory, MemoryPointer, TlsKey, HasMemory};
use memory::Kind as MemoryKind;
use operator;
use value::{PrimVal, PrimValKind, Value, Pointer};
use validation::ValidationQuery;

pub struct EvalContext<'a, 'tcx: 'a> {
    /// The results of the type checker, from rustc.
    pub(crate) tcx: TyCtxt<'a, 'tcx, 'tcx>,

    /// The virtual memory system.
    pub(crate) memory: Memory<'a, 'tcx>,

    #[allow(dead_code)]
    // FIXME(@RalfJung): validation branch
    /// Lvalues that were suspended by the validation subsystem, and will be recovered later
    pub(crate) suspended: HashMap<DynamicLifetime, Vec<ValidationQuery<'tcx>>>,

    /// Precomputed statics, constants and promoteds.
    pub(crate) globals: HashMap<GlobalId<'tcx>, Global<'tcx>>,

    /// The virtual call stack.
    pub(crate) stack: Vec<Frame<'tcx>>,

    /// The maximum number of stack frames allowed
    pub(crate) stack_limit: usize,

    /// The maximum number of operations that may be executed.
    /// This prevents infinite loops and huge computations from freezing up const eval.
    /// Remove once halting problem is solved.
    pub(crate) steps_remaining: u64,

    /// Environment variables set by `setenv`
    /// Miri does not expose env vars from the host to the emulated program
    pub(crate) env_vars: HashMap<Vec<u8>, MemoryPointer>,
}

/// A stack frame.
pub struct Frame<'tcx> {
    ////////////////////////////////////////////////////////////////////////////////
    // Function and callsite information
    ////////////////////////////////////////////////////////////////////////////////

    /// The MIR for the function called on this frame.
    pub mir: &'tcx mir::Mir<'tcx>,

    /// The def_id and substs of the current function
    pub instance: ty::Instance<'tcx>,

    /// The span of the call site.
    pub span: codemap::Span,

    ////////////////////////////////////////////////////////////////////////////////
    // Return lvalue and locals
    ////////////////////////////////////////////////////////////////////////////////

    /// The block to return to when returning from the current stack frame
    pub return_to_block: StackPopCleanup,

    /// The location where the result of the current stack frame should be written to.
    pub return_lvalue: Lvalue<'tcx>,

    /// The list of locals for this stack frame, stored in order as
    /// `[arguments..., variables..., temporaries...]`. The locals are stored as `Option<Value>`s.
    /// `None` represents a local that is currently dead, while a live local
    /// can either directly contain `PrimVal` or refer to some part of an `Allocation`.
    ///
    /// Before being initialized, arguments are `Value::ByVal(PrimVal::Undef)` and other locals are `None`.
    pub locals: Vec<Option<Value>>,

    ////////////////////////////////////////////////////////////////////////////////
    // Current position within the function
    ////////////////////////////////////////////////////////////////////////////////

    /// The block that is currently executed (or will be executed after the above call stacks
    /// return).
    pub block: mir::BasicBlock,

    /// The index of the currently evaluated statment.
    pub stmt: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum StackPopCleanup {
    /// The stackframe existed to compute the initial value of a static/constant, make sure it
    /// isn't modifyable afterwards in case of constants.
    /// In case of `static mut`, mark the memory to ensure it's never marked as immutable through
    /// references or deallocated
    MarkStatic(Mutability),
    /// A regular stackframe added due to a function call will need to get forwarded to the next
    /// block
    Goto(mir::BasicBlock),
    /// After finishing a tls destructor, find the next one instead of starting from the beginning
    /// and thus just rerunning the first one until its `data` argument is null
    ///
    /// The index is the current tls destructor's index
    Tls(Option<TlsKey>),
    /// The main function and diverging functions have nowhere to return to
    None,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicLifetime {
    pub frame: usize,
    pub region: Option<CodeExtent>, // "None" indicates "until the function ends"
}

#[derive(Copy, Clone, Debug)]
pub struct ResourceLimits {
    pub memory_size: u64,
    pub step_limit: u64,
    pub stack_limit: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        ResourceLimits {
            memory_size: 100 * 1024 * 1024, // 100 MB
            step_limit: 1_000_000,
            stack_limit: 100,
        }
    }
}

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>, limits: ResourceLimits) -> Self {
        EvalContext {
            tcx,
            memory: Memory::new(&tcx.data_layout, limits.memory_size),
            suspended: HashMap::new(),
            globals: HashMap::new(),
            stack: Vec::new(),
            stack_limit: limits.stack_limit,
            steps_remaining: limits.step_limit,
            env_vars: HashMap::new(),
        }
    }

    pub fn alloc_ptr(&mut self, ty: Ty<'tcx>) -> EvalResult<'tcx, MemoryPointer> {
        let substs = self.substs();
        self.alloc_ptr_with_substs(ty, substs)
    }

    pub fn alloc_ptr_with_substs(
        &mut self,
        ty: Ty<'tcx>,
        substs: &'tcx Substs<'tcx>
    ) -> EvalResult<'tcx, MemoryPointer> {
        let size = self.type_size_with_substs(ty, substs)?.expect("cannot alloc memory for unsized type");
        let align = self.type_align_with_substs(ty, substs)?;
        self.memory.allocate(size, align, MemoryKind::Stack)
    }

    pub fn memory(&self) -> &Memory<'a, 'tcx> {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut Memory<'a, 'tcx> {
        &mut self.memory
    }

    pub fn stack(&self) -> &[Frame<'tcx>] {
        &self.stack
    }

    #[inline]
    pub fn cur_frame(&self) -> usize {
        assert!(self.stack.len() > 0);
        self.stack.len() - 1
    }

    /// Returns true if the current frame or any parent frame is part of a ctfe.
    ///
    /// Used to disable features in const eval, which do not have a rfc enabling
    /// them or which can't be written in a way that they produce the same output
    /// that evaluating the code at runtime would produce.
    pub fn const_env(&self) -> bool {
        for frame in self.stack.iter().rev() {
            if let StackPopCleanup::MarkStatic(_) = frame.return_to_block {
                return true;
            }
        }
        false
    }

    pub(crate) fn str_to_value(&mut self, s: &str) -> EvalResult<'tcx, Value> {
        let ptr = self.memory.allocate_cached(s.as_bytes())?;
        Ok(Value::ByValPair(PrimVal::Ptr(ptr), PrimVal::from_u128(s.len() as u128)))
    }

    pub(super) fn const_to_value(&mut self, const_val: &ConstVal<'tcx>) -> EvalResult<'tcx, Value> {
        use rustc::middle::const_val::ConstVal::*;
        use rustc_const_math::ConstFloat;

        let primval = match *const_val {
            Integral(const_int) => PrimVal::Bytes(const_int.to_u128_unchecked()),

            Float(ConstFloat::F32(f)) => PrimVal::from_f32(f),
            Float(ConstFloat::F64(f)) => PrimVal::from_f64(f),

            Bool(b) => PrimVal::from_bool(b),
            Char(c) => PrimVal::from_char(c),

            Str(ref s) => return self.str_to_value(s),

            ByteStr(ref bs) => {
                let ptr = self.memory.allocate_cached(bs)?;
                PrimVal::Ptr(ptr)
            }

            Variant(_)   => unimplemented!(),
            Struct(_)    => unimplemented!(),
            Tuple(_)     => unimplemented!(),
            // function items are zero sized and thus have no readable value
            Function(..)  => PrimVal::Undef,
            Array(_)     => unimplemented!(),
            Repeat(_, _) => unimplemented!(),
        };

        Ok(Value::ByVal(primval))
    }

    pub(super) fn type_is_sized(&self, ty: Ty<'tcx>) -> bool {
        // generics are weird, don't run this function on a generic
        assert!(!ty.needs_subst());
        ty.is_sized(self.tcx, ty::ParamEnv::empty(Reveal::All), DUMMY_SP)
    }

    pub fn load_mir(&self, instance: ty::InstanceDef<'tcx>) -> EvalResult<'tcx, &'tcx mir::Mir<'tcx>> {
        trace!("load mir {:?}", instance);
        match instance {
            ty::InstanceDef::Item(def_id) => self.tcx.maybe_optimized_mir(def_id).ok_or_else(|| EvalError::NoMirFor(self.tcx.item_path_str(def_id))),
            _ => Ok(self.tcx.instance_mir(instance)),
        }
    }

    pub fn monomorphize(&self, ty: Ty<'tcx>, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        // miri doesn't care about lifetimes, and will choke on some crazy ones
        // let's simply get rid of them
        let without_lifetimes = self.tcx.erase_regions(&ty);
        let substituted = without_lifetimes.subst(self.tcx, substs);
        self.tcx.normalize_associated_type(&substituted)
    }

    pub fn erase_lifetimes<T>(&self, value: &Binder<T>) -> T
        where T : TypeFoldable<'tcx>
    {
        let value = self.tcx.erase_late_bound_regions(value);
        self.tcx.erase_regions(&value)
    }

    pub(super) fn type_size(&self, ty: Ty<'tcx>) -> EvalResult<'tcx, Option<u64>> {
        self.type_size_with_substs(ty, self.substs())
    }

    pub(super) fn type_align(&self, ty: Ty<'tcx>) -> EvalResult<'tcx, u64> {
        self.type_align_with_substs(ty, self.substs())
    }

    fn type_size_with_substs(
        &self,
        ty: Ty<'tcx>,
        substs: &'tcx Substs<'tcx>,
    ) -> EvalResult<'tcx, Option<u64>> {
        let layout = self.type_layout_with_substs(ty, substs)?;
        if layout.is_unsized() {
            Ok(None)
        } else {
            Ok(Some(layout.size(&self.tcx.data_layout).bytes()))
        }
    }

    fn type_align_with_substs(&self, ty: Ty<'tcx>, substs: &'tcx Substs<'tcx>) -> EvalResult<'tcx, u64> {
        self.type_layout_with_substs(ty, substs).map(|layout| layout.align(&self.tcx.data_layout).abi())
    }

    pub(super) fn type_layout(&self, ty: Ty<'tcx>) -> EvalResult<'tcx, &'tcx Layout> {
        self.type_layout_with_substs(ty, self.substs())
    }

    fn type_layout_with_substs(&self, ty: Ty<'tcx>, substs: &'tcx Substs<'tcx>) -> EvalResult<'tcx, &'tcx Layout> {
        // TODO(solson): Is this inefficient? Needs investigation.
        let ty = self.monomorphize(ty, substs);

        ty.layout(self.tcx, ty::ParamEnv::empty(Reveal::All)).map_err(EvalError::Layout)
    }

    pub fn push_stack_frame(
        &mut self,
        instance: ty::Instance<'tcx>,
        span: codemap::Span,
        mir: &'tcx mir::Mir<'tcx>,
        return_lvalue: Lvalue<'tcx>,
        return_to_block: StackPopCleanup,
    ) -> EvalResult<'tcx> {
        ::log_settings::settings().indentation += 1;

        /// Return the set of locals that have a storage annotation anywhere
        fn collect_storage_annotations<'tcx>(mir: &'tcx mir::Mir<'tcx>) -> HashSet<mir::Local> {
            use rustc::mir::StatementKind::*;

            let mut set = HashSet::new();
            for block in mir.basic_blocks() {
                for stmt in block.statements.iter() {
                    match stmt.kind {
                        StorageLive(mir::Lvalue::Local(local)) | StorageDead(mir::Lvalue::Local(local)) => {
                            set.insert(local);
                        }
                        _ => {}
                    }
                }
            };
            set
        }

        // Subtract 1 because `local_decls` includes the ReturnMemoryPointer, but we don't store a local
        // `Value` for that.
        let annotated_locals = collect_storage_annotations(mir);
        let num_locals = mir.local_decls.len() - 1;
        let mut locals = vec![None; num_locals];
        for i in 0..num_locals {
            let local = mir::Local::new(i+1);
            if !annotated_locals.contains(&local) {
                locals[i] = Some(Value::ByVal(PrimVal::Undef));
            }
        }

        self.stack.push(Frame {
            mir,
            block: mir::START_BLOCK,
            return_to_block,
            return_lvalue,
            locals,
            span,
            instance,
            stmt: 0,
        });

        let cur_frame = self.cur_frame();
        self.memory.set_cur_frame(cur_frame);

        if self.stack.len() > self.stack_limit {
            Err(EvalError::StackFrameLimitReached)
        } else {
            Ok(())
        }
    }

    pub(super) fn pop_stack_frame(&mut self) -> EvalResult<'tcx> {
        ::log_settings::settings().indentation -= 1;
        self.memory.locks_lifetime_ended(None);
        let frame = self.stack.pop().expect("tried to pop a stack frame, but there were none");
        if !self.stack.is_empty() {
            // TODO: IS this the correct time to start considering these accesses as originating from the returned-to stack frame?
            let cur_frame = self.cur_frame();
            self.memory.set_cur_frame(cur_frame);
        }
        match frame.return_to_block {
            StackPopCleanup::MarkStatic(mutable) => if let Lvalue::Global(id) = frame.return_lvalue {
                let global_value = self.globals.get_mut(&id)
                    .expect("global should have been cached (static)");
                match global_value.value {
                    // FIXME: to_ptr()? might be too extreme here, static zsts might reach this under certain conditions
                    Value::ByRef(ptr, _aligned) =>
                        // Alignment does not matter for this call
                        self.memory.mark_static_initalized(ptr.to_ptr()?.alloc_id, mutable)?,
                    Value::ByVal(val) => if let PrimVal::Ptr(ptr) = val {
                        self.memory.mark_inner_allocation(ptr.alloc_id, mutable)?;
                    },
                    Value::ByValPair(val1, val2) => {
                        if let PrimVal::Ptr(ptr) = val1 {
                            self.memory.mark_inner_allocation(ptr.alloc_id, mutable)?;
                        }
                        if let PrimVal::Ptr(ptr) = val2 {
                            self.memory.mark_inner_allocation(ptr.alloc_id, mutable)?;
                        }
                    },
                }
                // see comment on `initialized` field
                assert!(!global_value.initialized);
                global_value.initialized = true;
                assert_eq!(global_value.mutable, Mutability::Mutable);
                global_value.mutable = mutable;
            } else {
                bug!("StackPopCleanup::MarkStatic on: {:?}", frame.return_lvalue);
            },
            StackPopCleanup::Goto(target) => self.goto_block(target),
            StackPopCleanup::None => {},
            StackPopCleanup::Tls(key) => {
                // either fetch the next dtor or start new from the beginning, if any are left with a non-null data
                let dtor = match self.memory.fetch_tls_dtor(key)? {
                    dtor @ Some(_) => dtor,
                    None => self.memory.fetch_tls_dtor(None)?,
                };
                if let Some((instance, ptr, key)) = dtor {
                    trace!("Running TLS dtor {:?} on {:?}", instance, ptr);
                    // TODO: Potentially, this has to support all the other possible instances? See eval_fn_call in terminator/mod.rs
                    let mir = self.load_mir(instance.def)?;
                    self.push_stack_frame(
                        instance,
                        mir.span,
                        mir,
                        Lvalue::undef(),
                        StackPopCleanup::Tls(Some(key)),
                    )?;
                    let arg_local = self.frame().mir.args_iter().next().ok_or(EvalError::AbiViolation("TLS dtor does not take enough arguments.".to_owned()))?;
                    let dest = self.eval_lvalue(&mir::Lvalue::Local(arg_local))?;
                    let ty = self.tcx.mk_mut_ptr(self.tcx.types.u8);
                    self.write_ptr(dest, ptr, ty)?;
                }
            }
        }
        // deallocate all locals that are backed by an allocation
        for local in frame.locals {
            self.deallocate_local(local)?;
        }

        Ok(())
    }

    pub fn deallocate_local(&mut self, local: Option<Value>) -> EvalResult<'tcx> {
        if let Some(Value::ByRef(ptr, _aligned)) = local {
            trace!("deallocating local");
            let ptr = ptr.to_ptr()?;
            self.memory.dump_alloc(ptr.alloc_id);
            match self.memory.get(ptr.alloc_id)?.kind {
                // for a constant like `const FOO: &i32 = &1;` the local containing
                // the `1` is referred to by the global. We transitively marked everything
                // the global refers to as static itself, so we don't free it here
                MemoryKind::Static => {}
                MemoryKind::Stack => self.memory.deallocate(ptr, None, MemoryKind::Stack)?,
                other => bug!("local contained non-stack memory: {:?}", other),
            }
        };
        Ok(())
    }

    pub fn assign_discr_and_fields(
        &mut self,
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
        discr_offset: u64,
        operands: &[mir::Operand<'tcx>],
        discr_val: u128,
        variant_idx: usize,
        discr_size: u64,
    ) -> EvalResult<'tcx> {
        // FIXME(solson)
        let dest_ptr = self.force_allocation(dest)?.to_ptr()?;

        let discr_dest = dest_ptr.offset(discr_offset, &self)?;
        self.memory.write_uint(discr_dest, discr_val, discr_size)?;

        let dest = Lvalue::Ptr {
            ptr: dest_ptr.into(),
            extra: LvalueExtra::DowncastVariant(variant_idx),
            aligned: true,
        };

        self.assign_fields(dest, dest_ty, operands)
    }

    pub fn assign_fields(
        &mut self,
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
        operands: &[mir::Operand<'tcx>],
    ) -> EvalResult<'tcx> {
        if self.type_size(dest_ty)? == Some(0) {
            // zst assigning is a nop
            return Ok(());
        }
        if self.ty_to_primval_kind(dest_ty).is_ok() {
            assert_eq!(operands.len(), 1);
            let value = self.eval_operand(&operands[0])?;
            let value_ty = self.operand_ty(&operands[0]);
            return self.write_value(value, dest, value_ty);
        }
        for (field_index, operand) in operands.iter().enumerate() {
            let value = self.eval_operand(operand)?;
            let value_ty = self.operand_ty(operand);
            let field_dest = self.lvalue_field(dest, field_index, dest_ty, value_ty)?;
            self.write_value(value, field_dest, value_ty)?;
        }
        Ok(())
    }

    /// Evaluate an assignment statement.
    ///
    /// There is no separate `eval_rvalue` function. Instead, the code for handling each rvalue
    /// type writes its results directly into the memory specified by the lvalue.
    pub(super) fn eval_rvalue_into_lvalue(
        &mut self,
        rvalue: &mir::Rvalue<'tcx>,
        lvalue: &mir::Lvalue<'tcx>,
    ) -> EvalResult<'tcx> {
        let dest = self.eval_lvalue(lvalue)?;
        let dest_ty = self.lvalue_ty(lvalue);
        let dest_layout = self.type_layout(dest_ty)?;

        use rustc::mir::Rvalue::*;
        match *rvalue {
            Use(ref operand) => {
                let value = self.eval_operand(operand)?;
                self.write_value(value, dest, dest_ty)?;
            }

            BinaryOp(bin_op, ref left, ref right) => {
                if self.intrinsic_overflowing(bin_op, left, right, dest, dest_ty)? {
                    // There was an overflow in an unchecked binop.  Right now, we consider this an error and bail out.
                    // The rationale is that the reason rustc emits unchecked binops in release mode (vs. the checked binops
                    // it emits in debug mode) is performance, but it doesn't cost us any performance in miri.
                    // If, however, the compiler ever starts transforming unchecked intrinsics into unchecked binops,
                    // we have to go back to just ignoring the overflow here.
                    return Err(EvalError::OverflowingMath);
                }
            }

            CheckedBinaryOp(bin_op, ref left, ref right) => {
                self.intrinsic_with_overflow(bin_op, left, right, dest, dest_ty)?;
            }

            UnaryOp(un_op, ref operand) => {
                let val = self.eval_operand_to_primval(operand)?;
                let kind = self.ty_to_primval_kind(dest_ty)?;
                self.write_primval(dest, operator::unary_op(un_op, val, kind)?, dest_ty)?;
            }

            // Skip everything for zsts
            Aggregate(..) if self.type_size(dest_ty)? == Some(0) => {}

            Aggregate(ref kind, ref operands) => {
                self.inc_step_counter_and_check_limit(operands.len() as u64)?;
                use rustc::ty::layout::Layout::*;
                match *dest_layout {
                    Univariant { .. } | Array { .. } => {
                        self.assign_fields(dest, dest_ty, operands)?;
                    }

                    General { discr, ref variants, .. } => {
                        if let mir::AggregateKind::Adt(adt_def, variant, _, _) = **kind {
                            let discr_val = adt_def.discriminants(self.tcx)
                                .nth(variant)
                                .expect("broken mir: Adt variant id invalid")
                                .to_u128_unchecked();
                            let discr_size = discr.size().bytes();

                            self.assign_discr_and_fields(
                                dest,
                                dest_ty,
                                variants[variant].offsets[0].bytes(),
                                operands,
                                discr_val,
                                variant,
                                discr_size,
                            )?;
                        } else {
                            bug!("tried to assign {:?} to Layout::General", kind);
                        }
                    }

                    RawNullablePointer { nndiscr, .. } => {
                        if let mir::AggregateKind::Adt(_, variant, _, _) = **kind {
                            if nndiscr == variant as u64 {
                                assert_eq!(operands.len(), 1);
                                let operand = &operands[0];
                                let value = self.eval_operand(operand)?;
                                let value_ty = self.operand_ty(operand);
                                self.write_value(value, dest, value_ty)?;
                            } else {
                                if let Some(operand) = operands.get(0) {
                                    assert_eq!(operands.len(), 1);
                                    let operand_ty = self.operand_ty(operand);
                                    assert_eq!(self.type_size(operand_ty)?, Some(0));
                                }
                                self.write_null(dest, dest_ty)?;
                            }
                        } else {
                            bug!("tried to assign {:?} to Layout::RawNullablePointer", kind);
                        }
                    }

                    StructWrappedNullablePointer { nndiscr, ref discrfield, .. } => {
                        if let mir::AggregateKind::Adt(_, variant, _, _) = **kind {
                            if nndiscr == variant as u64 {
                                self.assign_fields(dest, dest_ty, operands)?;
                            } else {
                                for operand in operands {
                                    let operand_ty = self.operand_ty(operand);
                                    assert_eq!(self.type_size(operand_ty)?, Some(0));
                                }
                                let (offset, ty, _packed) = self.nonnull_offset_and_ty(dest_ty, nndiscr, discrfield)?;
                                // TODO: The packed flag is ignored

                                // FIXME(solson)
                                let dest = self.force_allocation(dest)?.to_ptr()?;

                                let dest = dest.offset(offset.bytes(), &self)?;
                                let dest_size = self.type_size(ty)?
                                    .expect("bad StructWrappedNullablePointer discrfield");
                                self.memory.write_int(dest, 0, dest_size)?;
                            }
                        } else {
                            bug!("tried to assign {:?} to Layout::RawNullablePointer", kind);
                        }
                    }

                    CEnum { .. } => {
                        assert_eq!(operands.len(), 0);
                        if let mir::AggregateKind::Adt(adt_def, variant, _, _) = **kind {
                            let n = adt_def.discriminants(self.tcx)
                                .nth(variant)
                                .expect("broken mir: Adt variant index invalid")
                                .to_u128_unchecked();
                            self.write_primval(dest, PrimVal::Bytes(n), dest_ty)?;
                        } else {
                            bug!("tried to assign {:?} to Layout::CEnum", kind);
                        }
                    }

                    Vector { count, .. } => {
                        debug_assert_eq!(count, operands.len() as u64);
                        self.assign_fields(dest, dest_ty, operands)?;
                    }

                    UntaggedUnion { .. } => {
                        assert_eq!(operands.len(), 1);
                        let operand = &operands[0];
                        let value = self.eval_operand(operand)?;
                        let value_ty = self.operand_ty(operand);
                        self.write_value(value, dest, value_ty)?;
                    }

                    _ => {
                        return Err(EvalError::Unimplemented(format!(
                            "can't handle destination layout {:?} when assigning {:?}",
                            dest_layout,
                            kind
                        )));
                    }
                }
            }

            Repeat(ref operand, _) => {
                let (elem_ty, length) = match dest_ty.sty {
                    ty::TyArray(elem_ty, n) => (elem_ty, n as u64),
                    _ => bug!("tried to assign array-repeat to non-array type {:?}", dest_ty),
                };
                self.inc_step_counter_and_check_limit(length)?;
                let elem_size = self.type_size(elem_ty)?
                    .expect("repeat element type must be sized");
                let value = self.eval_operand(operand)?;

                // FIXME(solson)
                let dest = Pointer::from(self.force_allocation(dest)?.to_ptr()?);

                for i in 0..length {
                    let elem_dest = dest.offset(i * elem_size, &self)?;
                    self.write_value_to_ptr(value, elem_dest, elem_ty)?;
                }
            }

            Len(ref lvalue) => {
                if self.const_env() {
                    return Err(EvalError::NeedsRfc("computing the length of arrays".to_string()));
                }
                let src = self.eval_lvalue(lvalue)?;
                let ty = self.lvalue_ty(lvalue);
                let (_, len) = src.elem_ty_and_len(ty);
                self.write_primval(dest, PrimVal::from_u128(len as u128), dest_ty)?;
            }

            Ref(_, _, ref lvalue) => {
                let src = self.eval_lvalue(lvalue)?;
                // We ignore the alignment of the lvalue here -- special handling for packed structs ends
                // at the `&` operator.
                let (ptr, extra, _aligned) = self.force_allocation(src)?.to_ptr_extra_aligned();

                let val = match extra {
                    LvalueExtra::None => ptr.to_value(),
                    LvalueExtra::Length(len) => ptr.to_value_with_len(len),
                    LvalueExtra::Vtable(vtable) => ptr.to_value_with_vtable(vtable),
                    LvalueExtra::DowncastVariant(..) =>
                        bug!("attempted to take a reference to an enum downcast lvalue"),
                };
                self.write_value(val, dest, dest_ty)?;
            }

            NullaryOp(mir::NullOp::Box, ty) => {
                if self.const_env() {
                    return Err(EvalError::NeedsRfc("\"heap\" allocations".to_string()));
                }
                // FIXME: call the `exchange_malloc` lang item if available
                let size = self.type_size(ty)?.expect("box only works with sized types");
                if size == 0 {
                    let align = self.type_align(ty)?;
                    self.write_primval(dest, PrimVal::Bytes(align.into()), dest_ty)?;
                } else {
                    let align = self.type_align(ty)?;
                    let ptr = self.memory.allocate(size, align, MemoryKind::Rust)?;
                    self.write_primval(dest, PrimVal::Ptr(ptr), dest_ty)?;
                }
            }

            NullaryOp(mir::NullOp::SizeOf, ty) => {
                if self.const_env() {
                    return Err(EvalError::NeedsRfc("computing the size of types (size_of)".to_string()));
                }
                let size = self.type_size(ty)?.expect("SizeOf nullary MIR operator called for unsized type");
                self.write_primval(dest, PrimVal::from_u128(size as u128), dest_ty)?;
            }

            Cast(kind, ref operand, cast_ty) => {
                debug_assert_eq!(self.monomorphize(cast_ty, self.substs()), dest_ty);
                use rustc::mir::CastKind::*;
                match kind {
                    Unsize => {
                        let src = self.eval_operand(operand)?;
                        let src_ty = self.operand_ty(operand);
                        self.unsize_into(src, src_ty, dest, dest_ty)?;
                    }

                    Misc => {
                        let src = self.eval_operand(operand)?;
                        let src_ty = self.operand_ty(operand);
                        if self.type_is_fat_ptr(src_ty) {
                            match (src, self.type_is_fat_ptr(dest_ty)) {
                                (Value::ByRef(..), _) |
                                (Value::ByValPair(..), true) => {
                                    self.write_value(src, dest, dest_ty)?;
                                },
                                (Value::ByValPair(data, _), false) => {
                                    self.write_value(Value::ByVal(data), dest, dest_ty)?;
                                },
                                (Value::ByVal(_), _) => bug!("expected fat ptr"),
                            }
                        } else {
                            let src_val = self.value_to_primval(src, src_ty)?;
                            let dest_val = self.cast_primval(src_val, src_ty, dest_ty)?;
                            self.write_value(Value::ByVal(dest_val), dest, dest_ty)?;
                        }
                    }

                    ReifyFnPointer => match self.operand_ty(operand).sty {
                        ty::TyFnDef(def_id, substs) => {
                            let instance = resolve(self.tcx, def_id, substs);
                            let fn_ptr = self.memory.create_fn_alloc(instance);
                            self.write_value(Value::ByVal(PrimVal::Ptr(fn_ptr)), dest, dest_ty)?;
                        },
                        ref other => bug!("reify fn pointer on {:?}", other),
                    },

                    UnsafeFnPointer => match dest_ty.sty {
                        ty::TyFnPtr(_) => {
                            let src = self.eval_operand(operand)?;
                            self.write_value(src, dest, dest_ty)?;
                        },
                        ref other => bug!("fn to unsafe fn cast on {:?}", other),
                    },

                    ClosureFnPointer => match self.operand_ty(operand).sty {
                        ty::TyClosure(def_id, substs) => {
                            let instance = resolve_closure(self.tcx, def_id, substs, ty::ClosureKind::FnOnce);
                            let fn_ptr = self.memory.create_fn_alloc(instance);
                            self.write_value(Value::ByVal(PrimVal::Ptr(fn_ptr)), dest, dest_ty)?;
                        },
                        ref other => bug!("closure fn pointer on {:?}", other),
                    },
                }
            }

            Discriminant(ref lvalue) => {
                let lval = self.eval_lvalue(lvalue)?;
                let ty = self.lvalue_ty(lvalue);
                let ptr = self.force_allocation(lval)?.to_ptr()?;
                let discr_val = self.read_discriminant_value(ptr, ty)?;
                if let ty::TyAdt(adt_def, _) = ty.sty {
                    if adt_def.discriminants(self.tcx).all(|v| discr_val != v.to_u128_unchecked()) {
                        return Err(EvalError::InvalidDiscriminant);
                    }
                } else {
                    bug!("rustc only generates Rvalue::Discriminant for enums");
                }
                self.write_primval(dest, PrimVal::Bytes(discr_val), dest_ty)?;
            },
        }

        if log_enabled!(::log::LogLevel::Trace) {
            self.dump_local(dest);
        }

        Ok(())
    }

    pub(super) fn type_is_fat_ptr(&self, ty: Ty<'tcx>) -> bool {
        match ty.sty {
            ty::TyRawPtr(ref tam) |
            ty::TyRef(_, ref tam) => !self.type_is_sized(tam.ty),
            ty::TyAdt(def, _) if def.is_box() => !self.type_is_sized(ty.boxed_ty()),
            _ => false,
        }
    }

    pub(super) fn nonnull_offset_and_ty(
        &self,
        ty: Ty<'tcx>,
        nndiscr: u64,
        discrfield: &[u32],
    ) -> EvalResult<'tcx, (Size, Ty<'tcx>, bool)> {
        // Skip the constant 0 at the start meant for LLVM GEP and the outer non-null variant
        let path = discrfield.iter().skip(2).map(|&i| i as usize);

        // Handle the field index for the outer non-null variant.
        let (inner_offset, inner_ty) = match ty.sty {
            ty::TyAdt(adt_def, substs) => {
                let variant = &adt_def.variants[nndiscr as usize];
                let index = discrfield[1];
                let field = &variant.fields[index as usize];
                (self.get_field_offset(ty, index as usize)?, field.ty(self.tcx, substs))
            }
            _ => bug!("non-enum for StructWrappedNullablePointer: {}", ty),
        };

        self.field_path_offset_and_ty(inner_offset, inner_ty, path)
    }

    fn field_path_offset_and_ty<I: Iterator<Item = usize>>(
        &self,
        mut offset: Size,
        mut ty: Ty<'tcx>,
        path: I,
    ) -> EvalResult<'tcx, (Size, Ty<'tcx>, bool)> {
        // Skip the initial 0 intended for LLVM GEP.
        let mut packed = false;
        for field_index in path {
            let field_offset = self.get_field_offset(ty, field_index)?;
            trace!("field_path_offset_and_ty: {}, {}, {:?}, {:?}", field_index, ty, field_offset, offset);
            let field_ty = self.get_field_ty(ty, field_index)?;
            ty = field_ty.0;
            packed = packed || field_ty.1;
            offset = offset.checked_add(field_offset, &self.tcx.data_layout).unwrap();
        }

        Ok((offset, ty, packed))
    }
    fn get_fat_field(&self, pointee_ty: Ty<'tcx>, field_index: usize) -> EvalResult<'tcx, Ty<'tcx>> {
        match (field_index, &self.tcx.struct_tail(pointee_ty).sty) {
            (1, &ty::TyStr) |
            (1, &ty::TySlice(_)) => Ok(self.tcx.types.usize),
            (1, &ty::TyDynamic(..)) |
            (0, _) => Ok(self.tcx.mk_imm_ptr(self.tcx.types.u8)),
            _ => bug!("invalid fat pointee type: {}", pointee_ty),
        }
    }

    /// Returns the field type and whether the field is packed
    pub fn get_field_ty(&self, ty: Ty<'tcx>, field_index: usize) -> EvalResult<'tcx, (Ty<'tcx>, bool)> {
        match ty.sty {
            ty::TyAdt(adt_def, _) if adt_def.is_box() =>
                Ok((self.get_fat_field(ty.boxed_ty(), field_index)?, false)),
            ty::TyAdt(adt_def, substs) if adt_def.is_enum() => {
                use rustc::ty::layout::Layout::*;
                match *self.type_layout(ty)? {
                    RawNullablePointer { nndiscr, .. } =>
                        Ok((adt_def.variants[nndiscr as usize].fields[field_index].ty(self.tcx, substs), false)),
                    StructWrappedNullablePointer { nndiscr, ref nonnull, .. } => {
                        let ty = adt_def.variants[nndiscr as usize].fields[field_index].ty(self.tcx, substs);
                        Ok((ty, nonnull.packed))
                    },
                    _ => Err(EvalError::Unimplemented(format!("get_field_ty can't handle enum type: {:?}, {:?}", ty, ty.sty))),
                }
            }
            ty::TyAdt(adt_def, substs) => {
                let variant_def = adt_def.struct_variant();
                use rustc::ty::layout::Layout::*;
                match *self.type_layout(ty)? {
                    Univariant { ref variant, .. } =>
                        Ok((variant_def.fields[field_index].ty(self.tcx, substs), variant.packed)),
                    _ => Err(EvalError::Unimplemented(format!("get_field_ty can't handle struct type: {:?}, {:?}", ty, ty.sty))),
                }
            }

            ty::TyTuple(fields, _) => Ok((fields[field_index], false)),

            ty::TyRef(_, ref tam) |
            ty::TyRawPtr(ref tam) => Ok((self.get_fat_field(tam.ty, field_index)?, false)),

            ty::TyArray(ref inner, _) => Ok((inner, false)),

            _ => Err(EvalError::Unimplemented(format!("can't handle type: {:?}, {:?}", ty, ty.sty))),
        }
    }

    fn get_field_offset(&self, ty: Ty<'tcx>, field_index: usize) -> EvalResult<'tcx, Size> {
        // Also see lvalue_field in lvalue.rs, which handles more cases but needs an actual value at the given type
        let layout = self.type_layout(ty)?;

        use rustc::ty::layout::Layout::*;
        match *layout {
            Univariant { ref variant, .. } => {
                Ok(variant.offsets[field_index])
            }
            FatPointer { .. } => {
                let bytes = field_index as u64 * self.memory.pointer_size();
                Ok(Size::from_bytes(bytes))
            }
            StructWrappedNullablePointer { ref nonnull, .. } => {
                Ok(nonnull.offsets[field_index])
            }
            _ => {
                let msg = format!("can't handle type: {:?}, with layout: {:?}", ty, layout);
                Err(EvalError::Unimplemented(msg))
            }
        }
    }

    pub fn get_field_count(&self, ty: Ty<'tcx>) -> EvalResult<'tcx, u64> {
        let layout = self.type_layout(ty)?;

        use rustc::ty::layout::Layout::*;
        match *layout {
            Univariant { ref variant, .. } => Ok(variant.offsets.len() as u64),
            FatPointer { .. } => Ok(2),
            StructWrappedNullablePointer { ref nonnull, .. } => Ok(nonnull.offsets.len() as u64),
            Vector { count , .. } |
            Array { count, .. } => Ok(count),
            Scalar { .. } => Ok(0),
            _ => {
                let msg = format!("can't handle type: {:?}, with layout: {:?}", ty, layout);
                Err(EvalError::Unimplemented(msg))
            }
        }
    }

    pub(super) fn wrapping_pointer_offset(&self, ptr: Pointer, pointee_ty: Ty<'tcx>, offset: i64) -> EvalResult<'tcx, Pointer> {
        // FIXME: assuming here that type size is < i64::max_value()
        let pointee_size = self.type_size(pointee_ty)?.expect("cannot offset a pointer to an unsized type") as i64;
        let offset = offset.overflowing_mul(pointee_size).0;
        ptr.wrapping_signed_offset(offset, self)
    }

    pub(super) fn pointer_offset(&self, ptr: Pointer, pointee_ty: Ty<'tcx>, offset: i64) -> EvalResult<'tcx, Pointer> {
        // This function raises an error if the offset moves the pointer outside of its allocation.  We consider
        // ZSTs their own huge allocation that doesn't overlap with anything (and nothing moves in there because the size is 0).
        // We also consider the NULL pointer its own separate allocation, and all the remaining integers pointers their own
        // allocation.

        if ptr.is_null()? { // NULL pointers must only be offset by 0
            return if offset == 0 { Ok(ptr) } else { Err(EvalError::InvalidNullPointerUsage) };
        }
        // FIXME: assuming here that type size is < i64::max_value()
        let pointee_size = self.type_size(pointee_ty)?.expect("cannot offset a pointer to an unsized type") as i64;
        return if let Some(offset) = offset.checked_mul(pointee_size) {
            let ptr = ptr.signed_offset(offset, self)?;
            // Do not do bounds-checking for integers; they can never alias a normal pointer anyway.
            if let PrimVal::Ptr(ptr) = ptr.into_inner_primval() {
                self.memory.check_bounds(ptr, false)?;
            } else if ptr.is_null()? {
                // We moved *to* a NULL pointer.  That seems wrong, LLVM considers the NULL pointer its own small allocation.  Reject this, for now.
                return Err(EvalError::InvalidNullPointerUsage);
            }
            Ok(ptr)
        } else {
            Err(EvalError::OverflowingMath)
        }
    }

    pub(super) fn eval_operand_to_primval(&mut self, op: &mir::Operand<'tcx>) -> EvalResult<'tcx, PrimVal> {
        let value = self.eval_operand(op)?;
        let ty = self.operand_ty(op);
        self.value_to_primval(value, ty)
    }

    pub(super) fn eval_operand(&mut self, op: &mir::Operand<'tcx>) -> EvalResult<'tcx, Value> {
        use rustc::mir::Operand::*;
        match *op {
            Consume(ref lvalue) => self.eval_and_read_lvalue(lvalue),

            Constant(ref constant) => {
                use rustc::mir::Literal;
                let mir::Constant { ref literal, .. } = **constant;
                let value = match *literal {
                    Literal::Value { ref value } => self.const_to_value(value)?,

                    Literal::Item { def_id, substs } => {
                        let instance = self.resolve_associated_const(def_id, substs);
                        let cid = GlobalId { instance, promoted: None };
                        self.globals.get(&cid).expect("static/const not cached").value
                    }

                    Literal::Promoted { index } => {
                        let cid = GlobalId {
                            instance: self.frame().instance,
                            promoted: Some(index),
                        };
                        self.globals.get(&cid).expect("promoted not cached").value
                    }
                };

                Ok(value)
            }
        }
    }

    pub(super) fn operand_ty(&self, operand: &mir::Operand<'tcx>) -> Ty<'tcx> {
        self.monomorphize(operand.ty(self.mir(), self.tcx), self.substs())
    }

    fn copy(&mut self, src: Pointer, dest: Pointer, ty: Ty<'tcx>) -> EvalResult<'tcx> {
        let size = self.type_size(ty)?.expect("cannot copy from an unsized type");
        let align = self.type_align(ty)?;
        self.memory.copy(src, dest, size, align, false)?;
        Ok(())
    }

    pub(super) fn force_allocation(
        &mut self,
        lvalue: Lvalue<'tcx>,
    ) -> EvalResult<'tcx, Lvalue<'tcx>> {
        let new_lvalue = match lvalue {
            Lvalue::Local { frame, local } => {
                // -1 since we don't store the return value
                match self.stack[frame].locals[local.index() - 1] {
                    None => return Err(EvalError::DeadLocal),
                    Some(Value::ByRef(ptr, aligned)) => {
                        Lvalue::Ptr { ptr, aligned, extra: LvalueExtra::None }
                    },
                    Some(val) => {
                        let ty = self.stack[frame].mir.local_decls[local].ty;
                        let ty = self.monomorphize(ty, self.stack[frame].instance.substs);
                        let substs = self.stack[frame].instance.substs;
                        let ptr = self.alloc_ptr_with_substs(ty, substs)?;
                        self.stack[frame].locals[local.index() - 1] = Some(Value::by_ref(ptr.into())); // it stays live
                        self.write_value_to_ptr(val, ptr.into(), ty)?;
                        Lvalue::from_ptr(ptr)
                    }
                }
            }
            Lvalue::Ptr { .. } => lvalue,
            Lvalue::Global(cid) => {
                let global_val = self.globals.get(&cid).expect("global not cached").clone();
                match global_val.value {
                    Value::ByRef(ptr, aligned) =>
                        Lvalue::Ptr { ptr, aligned, extra: LvalueExtra::None },
                    _ => {
                        let ptr = self.alloc_ptr_with_substs(global_val.ty, cid.instance.substs)?;
                        self.memory.mark_static(ptr.alloc_id);
                        self.write_value_to_ptr(global_val.value, ptr.into(), global_val.ty)?;
                        // see comment on `initialized` field
                        if global_val.initialized {
                            self.memory.mark_static_initalized(ptr.alloc_id, global_val.mutable)?;
                        }
                        let lval = self.globals.get_mut(&cid).expect("already checked");
                        *lval = Global {
                            value: Value::by_ref(ptr.into()),
                            .. global_val
                        };
                        Lvalue::from_ptr(ptr)
                    },
                }
            }
        };
        Ok(new_lvalue)
    }

    /// ensures this Value is not a ByRef
    pub(super) fn follow_by_ref_value(&mut self, value: Value, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        match value {
            Value::ByRef(ptr, aligned) => {
                self.read_maybe_aligned(aligned, |ectx| ectx.read_value(ptr, ty))
            }
            other => Ok(other),
        }
    }

    pub(super) fn value_to_primval(&mut self, value: Value, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        match self.follow_by_ref_value(value, ty)? {
            Value::ByRef(..) => bug!("follow_by_ref_value can't result in `ByRef`"),

            Value::ByVal(primval) => {
                self.ensure_valid_value(primval, ty)?;
                Ok(primval)
            }

            Value::ByValPair(..) => bug!("value_to_primval can't work with fat pointers"),
        }
    }

    pub(super) fn write_null(
        &mut self,
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        self.write_primval(dest, PrimVal::Bytes(0), dest_ty)
    }

    pub(super) fn write_ptr(
        &mut self,
        dest: Lvalue<'tcx>,
        val: Pointer,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        self.write_value(val.to_value(), dest, dest_ty)
    }

    pub(super) fn write_primval(
        &mut self,
        dest: Lvalue<'tcx>,
        val: PrimVal,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        self.write_value(Value::ByVal(val), dest, dest_ty)
    }

    pub(super) fn write_value(
        &mut self,
        src_val: Value,
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        //trace!("Writing {:?} to {:?} at type {:?}", src_val, dest, dest_ty);
        // Note that it is really important that the type here is the right one, and matches the type things are read at.
        // In case `src_val` is a `ByValPair`, we don't do any magic here to handle padding properly, which is only
        // correct if we never look at this data with the wrong type.

        match dest {
            Lvalue::Global(cid) => {
                let dest = self.globals.get_mut(&cid).expect("global should be cached").clone();
                if dest.mutable == Mutability::Immutable {
                    return Err(EvalError::ModifiedConstantMemory);
                }
                let write_dest = |this: &mut Self, val| {
                    *this.globals.get_mut(&cid).expect("already checked") = Global {
                        value: val,
                        ..dest
                    };
                    Ok(())
                };
                self.write_value_possibly_by_val(src_val, write_dest, dest.value, dest_ty)
            },

            Lvalue::Ptr { ptr, extra, aligned } => {
                assert_eq!(extra, LvalueExtra::None);
                self.write_maybe_aligned(aligned,
                    |ectx| ectx.write_value_to_ptr(src_val, ptr, dest_ty))
            }

            Lvalue::Local { frame, local } => {
                let dest = self.stack[frame].get_local(local)?;
                self.write_value_possibly_by_val(
                    src_val,
                    |this, val| this.stack[frame].set_local(local, val),
                    dest,
                    dest_ty,
                )
            }
        }
    }

    // The cases here can be a bit subtle. Read carefully!
    fn write_value_possibly_by_val<F: FnOnce(&mut Self, Value) -> EvalResult<'tcx>>(
        &mut self,
        src_val: Value,
        write_dest: F,
        old_dest_val: Value,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        if let Value::ByRef(dest_ptr, aligned) = old_dest_val {
            // If the value is already `ByRef` (that is, backed by an `Allocation`),
            // then we must write the new value into this allocation, because there may be
            // other pointers into the allocation. These other pointers are logically
            // pointers into the local variable, and must be able to observe the change.
            //
            // Thus, it would be an error to replace the `ByRef` with a `ByVal`, unless we
            // knew for certain that there were no outstanding pointers to this allocation.
            self.write_maybe_aligned(aligned,
                |ectx| ectx.write_value_to_ptr(src_val, dest_ptr, dest_ty))?;

        } else if let Value::ByRef(src_ptr, aligned) = src_val {
            // If the value is not `ByRef`, then we know there are no pointers to it
            // and we can simply overwrite the `Value` in the locals array directly.
            //
            // In this specific case, where the source value is `ByRef`, we must duplicate
            // the allocation, because this is a by-value operation. It would be incorrect
            // if they referred to the same allocation, since then a change to one would
            // implicitly change the other.
            //
            // It is a valid optimization to attempt reading a primitive value out of the
            // source and write that into the destination without making an allocation, so
            // we do so here.
            self.read_maybe_aligned(aligned, |ectx| {
                if let Ok(Some(src_val)) = ectx.try_read_value(src_ptr, dest_ty) {
                    write_dest(ectx, src_val)?;
                } else {
                    let dest_ptr = ectx.alloc_ptr(dest_ty)?.into();
                    ectx.copy(src_ptr, dest_ptr, dest_ty)?;
                    write_dest(ectx, Value::by_ref(dest_ptr))?;
                }
                Ok(())
            })?;

        } else {
            // Finally, we have the simple case where neither source nor destination are
            // `ByRef`. We may simply copy the source value over the the destintion.
            write_dest(self, src_val)?;
        }
        Ok(())
    }

    pub(super) fn write_value_to_ptr(
        &mut self,
        value: Value,
        dest: Pointer,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        match value {
            Value::ByRef(ptr, aligned) => {
                self.read_maybe_aligned(aligned, |ectx| ectx.copy(ptr, dest, dest_ty))
            },
            Value::ByVal(primval) => {
                let size = self.type_size(dest_ty)?.expect("dest type must be sized");
                self.memory.write_primval(dest, primval, size)
            }
            Value::ByValPair(a, b) => self.write_pair_to_ptr(a, b, dest.to_ptr()?, dest_ty),
        }
    }

    pub(super) fn write_pair_to_ptr(
        &mut self,
        a: PrimVal,
        b: PrimVal,
        ptr: MemoryPointer,
        mut ty: Ty<'tcx>
    ) -> EvalResult<'tcx> {
        let mut packed = false;
        while self.get_field_count(ty)? == 1 {
            let field = self.get_field_ty(ty, 0)?;
            ty = field.0;
            packed = packed || field.1;
        }
        assert_eq!(self.get_field_count(ty)?, 2);
        let field_0 = self.get_field_offset(ty, 0)?;
        let field_1 = self.get_field_offset(ty, 1)?;
        let field_0_ty = self.get_field_ty(ty, 0)?;
        let field_1_ty = self.get_field_ty(ty, 1)?;
        // The .1 components say whether the field is packed
        assert_eq!(field_0_ty.1, field_1_ty.1, "the two fields must agree on being packed");
        packed = packed || field_0_ty.1;
        let field_0_size = self.type_size(field_0_ty.0)?.expect("pair element type must be sized");
        let field_1_size = self.type_size(field_1_ty.0)?.expect("pair element type must be sized");
        let field_0_ptr = ptr.offset(field_0.bytes(), &self)?.into();
        let field_1_ptr = ptr.offset(field_1.bytes(), &self)?.into();
        self.write_maybe_aligned(!packed,
            |ectx| ectx.memory.write_primval(field_0_ptr, a, field_0_size))?;
        self.write_maybe_aligned(!packed,
            |ectx| ectx.memory.write_primval(field_1_ptr, b, field_1_size))?;
        Ok(())
    }

    pub fn ty_to_primval_kind(&self, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimValKind> {
        use syntax::ast::FloatTy;

        let kind = match ty.sty {
            ty::TyBool => PrimValKind::Bool,
            ty::TyChar => PrimValKind::Char,

            ty::TyInt(int_ty) => {
                use syntax::ast::IntTy::*;
                let size = match int_ty {
                    I8 => 1,
                    I16 => 2,
                    I32 => 4,
                    I64 => 8,
                    I128 => 16,
                    Is => self.memory.pointer_size(),
                };
                PrimValKind::from_int_size(size)
            }

            ty::TyUint(uint_ty) => {
                use syntax::ast::UintTy::*;
                let size = match uint_ty {
                    U8 => 1,
                    U16 => 2,
                    U32 => 4,
                    U64 => 8,
                    U128 => 16,
                    Us => self.memory.pointer_size(),
                };
                PrimValKind::from_uint_size(size)
            }

            ty::TyFloat(FloatTy::F32) => PrimValKind::F32,
            ty::TyFloat(FloatTy::F64) => PrimValKind::F64,

            ty::TyFnPtr(_) => PrimValKind::FnPtr,

            ty::TyRef(_, ref tam) |
            ty::TyRawPtr(ref tam) if self.type_is_sized(tam.ty) => PrimValKind::Ptr,

            ty::TyAdt(def, _) if def.is_box() => PrimValKind::Ptr,

            ty::TyAdt(def, substs) => {
                use rustc::ty::layout::Layout::*;
                match *self.type_layout(ty)? {
                    CEnum { discr, signed, .. } => {
                        let size = discr.size().bytes();
                        if signed {
                            PrimValKind::from_int_size(size)
                        } else {
                            PrimValKind::from_uint_size(size)
                        }
                    }

                    RawNullablePointer { value, .. } => {
                        use rustc::ty::layout::Primitive::*;
                        match value {
                            // TODO(solson): Does signedness matter here? What should the sign be?
                            Int(int) => PrimValKind::from_uint_size(int.size().bytes()),
                            F32 => PrimValKind::F32,
                            F64 => PrimValKind::F64,
                            Pointer => PrimValKind::Ptr,
                        }
                    }

                    // represent single field structs as their single field
                    Univariant { .. } => {
                        // enums with just one variant are no different, but `.struct_variant()` doesn't work for enums
                        let variant = &def.variants[0];
                        // FIXME: also allow structs with only a single non zst field
                        if variant.fields.len() == 1 {
                            return self.ty_to_primval_kind(variant.fields[0].ty(self.tcx, substs));
                        } else {
                            return Err(EvalError::TypeNotPrimitive(ty));
                        }
                    }

                    _ => return Err(EvalError::TypeNotPrimitive(ty)),
                }
            }

            _ => return Err(EvalError::TypeNotPrimitive(ty)),
        };

        Ok(kind)
    }

    fn ensure_valid_value(&self, val: PrimVal, ty: Ty<'tcx>) -> EvalResult<'tcx> {
        match ty.sty {
            ty::TyBool if val.to_bytes()? > 1 => Err(EvalError::InvalidBool),

            ty::TyChar if ::std::char::from_u32(val.to_bytes()? as u32).is_none()
                => Err(EvalError::InvalidChar(val.to_bytes()? as u32 as u128)),

            _ => Ok(()),
        }
    }

    pub(super) fn read_value(&mut self, ptr: Pointer, ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        if let Some(val) = self.try_read_value(ptr, ty)? {
            Ok(val)
        } else {
            bug!("primitive read failed for type: {:?}", ty);
        }
    }

    pub(crate) fn read_ptr(&self, ptr: MemoryPointer, pointee_ty: Ty<'tcx>) -> EvalResult<'tcx, Value> {
        let p = self.memory.read_ptr(ptr)?;
        if self.type_is_sized(pointee_ty) {
            Ok(p.to_value())
        } else {
            trace!("reading fat pointer extra of type {}", pointee_ty);
            let extra = ptr.offset(self.memory.pointer_size(), self)?;
            match self.tcx.struct_tail(pointee_ty).sty {
                ty::TyDynamic(..) => Ok(p.to_value_with_vtable(self.memory.read_ptr(extra)?.to_ptr()?)),
                ty::TySlice(..) |
                ty::TyStr => Ok(p.to_value_with_len(self.memory.read_usize(extra)?)),
                _ => bug!("unsized primval ptr read from {:?}", pointee_ty),
            }
        }
    }

    fn try_read_value(&mut self, ptr: Pointer, ty: Ty<'tcx>) -> EvalResult<'tcx, Option<Value>> {
        use syntax::ast::FloatTy;

        let val = match ty.sty {
            ty::TyBool => PrimVal::from_bool(self.memory.read_bool(ptr.to_ptr()?)?),
            ty::TyChar => {
                let c = self.memory.read_uint(ptr.to_ptr()?, 4)? as u32;
                match ::std::char::from_u32(c) {
                    Some(ch) => PrimVal::from_char(ch),
                    None => return Err(EvalError::InvalidChar(c as u128)),
                }
            }

            ty::TyInt(int_ty) => {
                use syntax::ast::IntTy::*;
                let size = match int_ty {
                    I8 => 1,
                    I16 => 2,
                    I32 => 4,
                    I64 => 8,
                    I128 => 16,
                    Is => self.memory.pointer_size(),
                };
                // if we transmute a ptr to an isize, reading it back into a primval shouldn't panic
                // Due to read_ptr ignoring the sign, we need to jump around some hoops
                match self.memory.read_int(ptr.to_ptr()?, size) {
                    Err(EvalError::ReadPointerAsBytes) if size == self.memory.pointer_size() =>
                        // Reading as an int failed because we are seeing ptr bytes *and* we are actually reading at ptr size.
                        // Let's try again, reading a ptr this time.
                        self.memory.read_ptr(ptr.to_ptr()?)?.into_inner_primval(),
                    other => PrimVal::from_i128(other?),
                }
            }

            ty::TyUint(uint_ty) => {
                use syntax::ast::UintTy::*;
                let size = match uint_ty {
                    U8 => 1,
                    U16 => 2,
                    U32 => 4,
                    U64 => 8,
                    U128 => 16,
                    Us => self.memory.pointer_size(),
                };
                // if we transmute a ptr to an usize, reading it back into a primval shouldn't panic
                // for consistency's sake, we use the same code as above
                match self.memory.read_uint(ptr.to_ptr()?, size) {
                    Err(EvalError::ReadPointerAsBytes) if size == self.memory.pointer_size() => self.memory.read_ptr(ptr.to_ptr()?)?.into_inner_primval(),
                    other => PrimVal::from_u128(other?),
                }
            }

            ty::TyFloat(FloatTy::F32) => PrimVal::from_f32(self.memory.read_f32(ptr.to_ptr()?)?),
            ty::TyFloat(FloatTy::F64) => PrimVal::from_f64(self.memory.read_f64(ptr.to_ptr()?)?),

            ty::TyFnPtr(_) => self.memory.read_ptr(ptr.to_ptr()?)?.into_inner_primval(),
            ty::TyRef(_, ref tam) |
            ty::TyRawPtr(ref tam) => return self.read_ptr(ptr.to_ptr()?, tam.ty).map(Some),

            ty::TyAdt(def, _) => {
                if def.is_box() {
                    return self.read_ptr(ptr.to_ptr()?, ty.boxed_ty()).map(Some);
                }
                use rustc::ty::layout::Layout::*;
                if let CEnum { discr, signed, .. } = *self.type_layout(ty)? {
                    let size = discr.size().bytes();
                    if signed {
                        PrimVal::from_i128(self.memory.read_int(ptr.to_ptr()?, size)?)
                    } else {
                        PrimVal::from_u128(self.memory.read_uint(ptr.to_ptr()?, size)?)
                    }
                } else {
                    return Ok(None);
                }
            },

            _ => return Ok(None),
        };

        Ok(Some(Value::ByVal(val)))
    }

    pub(super) fn frame(&self) -> &Frame<'tcx> {
        self.stack.last().expect("no call frames exist")
    }

    pub(super) fn frame_mut(&mut self) -> &mut Frame<'tcx> {
        self.stack.last_mut().expect("no call frames exist")
    }

    pub(super) fn mir(&self) -> &'tcx mir::Mir<'tcx> {
        self.frame().mir
    }

    pub(super) fn substs(&self) -> &'tcx Substs<'tcx> {
        self.frame().instance.substs
    }

    fn unsize_into_ptr(
        &mut self,
        src: Value,
        src_ty: Ty<'tcx>,
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
        sty: Ty<'tcx>,
        dty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        // A<Struct> -> A<Trait> conversion
        let (src_pointee_ty, dest_pointee_ty) = self.tcx.struct_lockstep_tails(sty, dty);

        match (&src_pointee_ty.sty, &dest_pointee_ty.sty) {
            (&ty::TyArray(_, length), &ty::TySlice(_)) => {
                let ptr = src.into_ptr(&mut self.memory)?;
                // u64 cast is from usize to u64, which is always good
                self.write_value(ptr.to_value_with_len(length as u64), dest, dest_ty)
            }
            (&ty::TyDynamic(..), &ty::TyDynamic(..)) => {
                // For now, upcasts are limited to changes in marker
                // traits, and hence never actually require an actual
                // change to the vtable.
                self.write_value(src, dest, dest_ty)
            },
            (_, &ty::TyDynamic(ref data, _)) => {
                let trait_ref = data.principal().unwrap().with_self_ty(self.tcx, src_pointee_ty);
                let trait_ref = self.tcx.erase_regions(&trait_ref);
                let vtable = self.get_vtable(src_pointee_ty, trait_ref)?;
                let ptr = src.into_ptr(&mut self.memory)?;
                self.write_value(ptr.to_value_with_vtable(vtable), dest, dest_ty)
            },

            _ => bug!("invalid unsizing {:?} -> {:?}", src_ty, dest_ty),
        }
    }

    fn unsize_into(
        &mut self,
        src: Value,
        src_ty: Ty<'tcx>,
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        match (&src_ty.sty, &dest_ty.sty) {
            (&ty::TyRef(_, ref s), &ty::TyRef(_, ref d)) |
            (&ty::TyRef(_, ref s), &ty::TyRawPtr(ref d)) |
            (&ty::TyRawPtr(ref s), &ty::TyRawPtr(ref d)) => self.unsize_into_ptr(src, src_ty, dest, dest_ty, s.ty, d.ty),
            (&ty::TyAdt(def_a, substs_a), &ty::TyAdt(def_b, substs_b)) => {
                if def_a.is_box() || def_b.is_box() {
                    if !def_a.is_box() || !def_b.is_box() {
                        panic!("invalid unsizing between {:?} -> {:?}", src_ty, dest_ty);
                    }
                    return self.unsize_into_ptr(src, src_ty, dest, dest_ty, src_ty.boxed_ty(), dest_ty.boxed_ty());
                }
                if self.ty_to_primval_kind(src_ty).is_ok() {
                    // TODO: We ignore the packed flag here
                    let sty = self.get_field_ty(src_ty, 0)?.0;
                    let dty = self.get_field_ty(dest_ty, 0)?.0;
                    return self.unsize_into(src, sty, dest, dty);
                }
                // unsizing of generic struct with pointer fields
                // Example: `Arc<T>` -> `Arc<Trait>`
                // here we need to increase the size of every &T thin ptr field to a fat ptr

                assert_eq!(def_a, def_b);

                let src_fields = def_a.variants[0].fields.iter();
                let dst_fields = def_b.variants[0].fields.iter();

                //let src = adt::MaybeSizedValue::sized(src);
                //let dst = adt::MaybeSizedValue::sized(dst);
                let src_ptr = match src {
                    Value::ByRef(ptr, true) => ptr,
                    // TODO: Is it possible for unaligned pointers to occur here?
                    _ => bug!("expected aligned pointer, got {:?}", src),
                };

                // FIXME(solson)
                let dest = self.force_allocation(dest)?.to_ptr()?;
                let iter = src_fields.zip(dst_fields).enumerate();
                for (i, (src_f, dst_f)) in iter {
                    let src_fty = monomorphize_field_ty(self.tcx, src_f, substs_a);
                    let dst_fty = monomorphize_field_ty(self.tcx, dst_f, substs_b);
                    if self.type_size(dst_fty)? == Some(0) {
                        continue;
                    }
                    let src_field_offset = self.get_field_offset(src_ty, i)?.bytes();
                    let dst_field_offset = self.get_field_offset(dest_ty, i)?.bytes();
                    let src_f_ptr = src_ptr.offset(src_field_offset, &self)?;
                    let dst_f_ptr = dest.offset(dst_field_offset, &self)?;
                    if src_fty == dst_fty {
                        self.copy(src_f_ptr, dst_f_ptr.into(), src_fty)?;
                    } else {
                        self.unsize_into(Value::by_ref(src_f_ptr), src_fty, Lvalue::from_ptr(dst_f_ptr), dst_fty)?;
                    }
                }
                Ok(())
            }
            _ => bug!("unsize_into: invalid conversion: {:?} -> {:?}", src_ty, dest_ty),
        }
    }

    pub(super) fn dump_local(&self, lvalue: Lvalue<'tcx>) {
        // Debug output
        if let Lvalue::Local { frame, local } = lvalue {
            let mut allocs = Vec::new();
            let mut msg = format!("{:?}", local);
            if frame != self.cur_frame() {
                write!(msg, " ({} frames up)", self.cur_frame() - frame).unwrap();
            }
            write!(msg, ":").unwrap();

            match self.stack[frame].get_local(local) {
                Err(EvalError::DeadLocal) => {
                    write!(msg, " is dead").unwrap();
                }
                Err(err) => {
                    panic!("Failed to access local: {:?}", err);
                }
                Ok(Value::ByRef(ptr, aligned)) => match ptr.into_inner_primval() {
                    PrimVal::Ptr(ptr) => {
                        write!(msg, " by {}ref:", if aligned { "" } else { "unaligned " }).unwrap();
                        allocs.push(ptr.alloc_id);
                    },
                    ptr => write!(msg, " integral by ref: {:?}", ptr).unwrap(),
                },
                Ok(Value::ByVal(val)) => {
                    write!(msg, " {:?}", val).unwrap();
                    if let PrimVal::Ptr(ptr) = val { allocs.push(ptr.alloc_id); }
                }
                Ok(Value::ByValPair(val1, val2)) => {
                    write!(msg, " ({:?}, {:?})", val1, val2).unwrap();
                    if let PrimVal::Ptr(ptr) = val1 { allocs.push(ptr.alloc_id); }
                    if let PrimVal::Ptr(ptr) = val2 { allocs.push(ptr.alloc_id); }
                }
            }

            trace!("{}", msg);
            self.memory.dump_allocs(allocs);
        }
    }

    /// Convenience function to ensure correct usage of globals and code-sharing with locals.
    pub fn modify_global<F>(&mut self, cid: GlobalId<'tcx>, f: F) -> EvalResult<'tcx>
        where F: FnOnce(&mut Self, Value) -> EvalResult<'tcx, Value>,
    {
        let mut val = self.globals.get(&cid).expect("global not cached").clone();
        if val.mutable == Mutability::Immutable {
            return Err(EvalError::ModifiedConstantMemory);
        }
        val.value = f(self, val.value)?;
        *self.globals.get_mut(&cid).expect("already checked") = val;
        Ok(())
    }

    /// Convenience function to ensure correct usage of locals and code-sharing with globals.
    pub fn modify_local<F>(
        &mut self,
        frame: usize,
        local: mir::Local,
        f: F,
    ) -> EvalResult<'tcx>
        where F: FnOnce(&mut Self, Value) -> EvalResult<'tcx, Value>,
    {
        let val = self.stack[frame].get_local(local)?;
        let new_val = f(self, val)?;
        self.stack[frame].set_local(local, new_val)?;
        // FIXME(solson): Run this when setting to Undef? (See previous version of this code.)
        // if let Value::ByRef(ptr) = self.stack[frame].get_local(local) {
        //     self.memory.deallocate(ptr)?;
        // }
        Ok(())
    }
}

impl<'tcx> Frame<'tcx> {
    pub fn get_local(&self, local: mir::Local) -> EvalResult<'tcx, Value> {
        // Subtract 1 because we don't store a value for the ReturnPointer, the local with index 0.
        self.locals[local.index() - 1].ok_or(EvalError::DeadLocal)
    }

    fn set_local(&mut self, local: mir::Local, value: Value) -> EvalResult<'tcx> {
        // Subtract 1 because we don't store a value for the ReturnPointer, the local with index 0.
        match self.locals[local.index() - 1] {
            None => Err(EvalError::DeadLocal),
            Some(ref mut local) => {
                *local = value;
                Ok(())
            }
        }
    }

    pub fn storage_live(&mut self, local: mir::Local) -> EvalResult<'tcx, Option<Value>> {
        trace!("{:?} is now live", local);

        let old = self.locals[local.index() - 1];
        self.locals[local.index() - 1] = Some(Value::ByVal(PrimVal::Undef)); // StorageLive *always* kills the value that's currently stored
        return Ok(old);
    }

    /// Returns the old value of the local
    pub fn storage_dead(&mut self, local: mir::Local) -> EvalResult<'tcx, Option<Value>> {
        trace!("{:?} is now dead", local);

        let old = self.locals[local.index() - 1];
        self.locals[local.index() - 1] = None;
        return Ok(old);
    }
}

pub fn eval_main<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    main_id: DefId,
    start_wrapper: Option<DefId>,
    limits: ResourceLimits,
) {
    fn run_main<'a, 'tcx: 'a>(
        ecx: &mut EvalContext<'a, 'tcx>,
        main_id: DefId,
        start_wrapper: Option<DefId>,
    ) -> EvalResult<'tcx> {
        let main_instance = ty::Instance::mono(ecx.tcx, main_id);
        let main_mir = ecx.load_mir(main_instance.def)?;
        let mut cleanup_ptr = None; // Pointer to be deallocated when we are done

        if !main_mir.return_ty.is_nil() || main_mir.arg_count != 0 {
            return Err(EvalError::Unimplemented("miri does not support main functions without `fn()` type signatures".to_owned()));
        }

        if let Some(start_id) = start_wrapper {
            let start_instance = ty::Instance::mono(ecx.tcx, start_id);
            let start_mir = ecx.load_mir(start_instance.def)?;

            if start_mir.arg_count != 3 {
                return Err(EvalError::AbiViolation(format!("'start' lang item should have three arguments, but has {}", start_mir.arg_count)));
            }

            // Return value
            let ret_ptr = ecx.memory.allocate(ecx.tcx.data_layout.pointer_size.bytes(), ecx.tcx.data_layout.pointer_align.abi(), MemoryKind::Stack)?;
            cleanup_ptr = Some(ret_ptr);

            // Push our stack frame
            ecx.push_stack_frame(
                start_instance,
                start_mir.span,
                start_mir,
                Lvalue::from_ptr(ret_ptr),
                StackPopCleanup::Tls(None),
            )?;

            let mut args = ecx.frame().mir.args_iter();

            // First argument: pointer to main()
            let main_ptr = ecx.memory.create_fn_alloc(main_instance);
            let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
            let main_ty = main_instance.def.def_ty(ecx.tcx);
            let main_ptr_ty = ecx.tcx.mk_fn_ptr(main_ty.fn_sig(ecx.tcx));
            ecx.write_value(Value::ByVal(PrimVal::Ptr(main_ptr)), dest, main_ptr_ty)?;

            // Second argument (argc): 0
            let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
            let ty = ecx.tcx.types.isize;
            ecx.write_null(dest, ty)?;

            // Third argument (argv): 0
            let dest = ecx.eval_lvalue(&mir::Lvalue::Local(args.next().unwrap()))?;
            let ty = ecx.tcx.mk_imm_ptr(ecx.tcx.mk_imm_ptr(ecx.tcx.types.u8));
            ecx.write_null(dest, ty)?;
        } else {
            ecx.push_stack_frame(
                main_instance,
                main_mir.span,
                main_mir,
                Lvalue::undef(),
                StackPopCleanup::Tls(None),
            )?;
        }

        while ecx.step()? {}
        if let Some(cleanup_ptr) = cleanup_ptr {
            ecx.memory.deallocate(cleanup_ptr, None, MemoryKind::Stack)?;
        }
        return Ok(());
    }

    let mut ecx = EvalContext::new(tcx, limits);
    match run_main(&mut ecx, main_id, start_wrapper) {
        Ok(()) => {
            let leaks = ecx.memory.leak_report();
            if leaks != 0 {
                tcx.sess.err("the evaluated program leaked memory");
            }
        }
        Err(e) => {
            report(tcx, &ecx, &e);
        }
    }
}

fn report(tcx: TyCtxt, ecx: &EvalContext, e: &EvalError) {
    if let Some(frame) = ecx.stack().last() {
        let block = &frame.mir.basic_blocks()[frame.block];
        let span = if frame.stmt < block.statements.len() {
            block.statements[frame.stmt].source_info.span
        } else {
            block.terminator().source_info.span
        };
        let mut err = tcx.sess.struct_span_err(span, &e.to_string());
        for &Frame { instance, span, .. } in ecx.stack().iter().rev() {
            if tcx.def_key(instance.def_id()).disambiguated_data.data == DefPathData::ClosureExpr {
                err.span_note(span, "inside call to closure");
                continue;
            }
            err.span_note(span, &format!("inside call to {}", instance));
        }
        err.emit();
    } else {
        tcx.sess.err(&e.to_string());
    }
}

// TODO(solson): Upstream these methods into rustc::ty::layout.

pub(super) trait IntegerExt {
    fn size(self) -> Size;
}

impl IntegerExt for layout::Integer {
    fn size(self) -> Size {
        use rustc::ty::layout::Integer::*;
        match self {
            I1 | I8 => Size::from_bits(8),
            I16 => Size::from_bits(16),
            I32 => Size::from_bits(32),
            I64 => Size::from_bits(64),
            I128 => Size::from_bits(128),
        }
    }
}


pub fn monomorphize_field_ty<'a, 'tcx:'a >(tcx: TyCtxt<'a, 'tcx, 'tcx>, f: &ty::FieldDef, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
    let substituted = f.ty(tcx, substs);
    tcx.normalize_associated_type(&substituted)
}

pub fn is_inhabited<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> bool {
    ty.uninhabited_from(&mut HashMap::default(), tcx).is_empty()
}

/// FIXME: expose trans::monomorphize::resolve_closure
pub fn resolve_closure<'a, 'tcx> (
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    def_id: DefId,
    substs: ty::ClosureSubsts<'tcx>,
    requested_kind: ty::ClosureKind,
) -> ty::Instance<'tcx> {
    let actual_kind = tcx.closure_kind(def_id);
    match needs_fn_once_adapter_shim(actual_kind, requested_kind) {
        Ok(true) => fn_once_adapter_instance(tcx, def_id, substs),
        _ => ty::Instance::new(def_id, substs.substs)
    }
}

fn fn_once_adapter_instance<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    closure_did: DefId,
    substs: ty::ClosureSubsts<'tcx>,
) -> ty::Instance<'tcx> {
    debug!("fn_once_adapter_shim({:?}, {:?})",
           closure_did,
           substs);
    let fn_once = tcx.lang_items.fn_once_trait().unwrap();
    let call_once = tcx.associated_items(fn_once)
        .find(|it| it.kind == ty::AssociatedKind::Method)
        .unwrap().def_id;
    let def = ty::InstanceDef::ClosureOnceShim { call_once };

    let self_ty = tcx.mk_closure_from_closure_substs(
        closure_did, substs);

    let sig = tcx.fn_sig(closure_did).subst(tcx, substs.substs);
    let sig = tcx.erase_late_bound_regions_and_normalize(&sig);
    assert_eq!(sig.inputs().len(), 1);
    let substs = tcx.mk_substs([
        Kind::from(self_ty),
        Kind::from(sig.inputs()[0]),
    ].iter().cloned());

    debug!("fn_once_adapter_shim: self_ty={:?} sig={:?}", self_ty, sig);
    ty::Instance { def, substs }
}

fn needs_fn_once_adapter_shim(actual_closure_kind: ty::ClosureKind,
                              trait_closure_kind: ty::ClosureKind)
                              -> Result<bool, ()>
{
    match (actual_closure_kind, trait_closure_kind) {
        (ty::ClosureKind::Fn, ty::ClosureKind::Fn) |
        (ty::ClosureKind::FnMut, ty::ClosureKind::FnMut) |
        (ty::ClosureKind::FnOnce, ty::ClosureKind::FnOnce) => {
            // No adapter needed.
           Ok(false)
        }
        (ty::ClosureKind::Fn, ty::ClosureKind::FnMut) => {
            // The closure fn `llfn` is a `fn(&self, ...)`.  We want a
            // `fn(&mut self, ...)`. In fact, at trans time, these are
            // basically the same thing, so we can just return llfn.
            Ok(false)
        }
        (ty::ClosureKind::Fn, ty::ClosureKind::FnOnce) |
        (ty::ClosureKind::FnMut, ty::ClosureKind::FnOnce) => {
            // The closure fn `llfn` is a `fn(&self, ...)` or `fn(&mut
            // self, ...)`.  We want a `fn(self, ...)`. We can produce
            // this by doing something like:
            //
            //     fn call_once(self, ...) { call_mut(&self, ...) }
            //     fn call_once(mut self, ...) { call_mut(&mut self, ...) }
            //
            // These are both the same at trans time.
            Ok(true)
        }
        _ => Err(()),
    }
}

/// The point where linking happens. Resolve a (def_id, substs)
/// pair to an instance.
pub fn resolve<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    def_id: DefId,
    substs: &'tcx Substs<'tcx>
) -> ty::Instance<'tcx> {
    debug!("resolve(def_id={:?}, substs={:?})",
           def_id, substs);
    let result = if let Some(trait_def_id) = tcx.trait_of_item(def_id) {
        debug!(" => associated item, attempting to find impl");
        let item = tcx.associated_item(def_id);
        resolve_associated_item(tcx, &item, trait_def_id, substs)
    } else {
        let item_type = def_ty(tcx, def_id, substs);
        let def = match item_type.sty {
            ty::TyFnDef(..) if {
                    let f = item_type.fn_sig(tcx);
                    f.abi() == Abi::RustIntrinsic ||
                    f.abi() == Abi::PlatformIntrinsic
                } =>
            {
                debug!(" => intrinsic");
                ty::InstanceDef::Intrinsic(def_id)
            }
            _ => {
                if Some(def_id) == tcx.lang_items.drop_in_place_fn() {
                    let ty = substs.type_at(0);
                    if needs_drop_glue(tcx, ty) {
                        debug!(" => nontrivial drop glue");
                        ty::InstanceDef::DropGlue(def_id, Some(ty))
                    } else {
                        debug!(" => trivial drop glue");
                        ty::InstanceDef::DropGlue(def_id, None)
                    }
                } else {
                    debug!(" => free item");
                    ty::InstanceDef::Item(def_id)
                }
            }
        };
        ty::Instance { def, substs }
    };
    debug!("resolve(def_id={:?}, substs={:?}) = {}",
           def_id, substs, result);
    result
}

pub fn needs_drop_glue<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, t: Ty<'tcx>) -> bool {
    assert!(t.is_normalized_for_trans());

    let t = tcx.erase_regions(&t);

    // FIXME (#22815): note that type_needs_drop conservatively
    // approximates in some cases and may say a type expression
    // requires drop glue when it actually does not.
    //
    // (In this case it is not clear whether any harm is done, i.e.
    // erroneously returning `true` in some cases where we could have
    // returned `false` does not appear unsound. The impact on
    // code quality is unknown at this time.)

    let env = ty::ParamEnv::empty(Reveal::All);
    if !t.needs_drop(tcx, env) {
        return false;
    }
    match t.sty {
        ty::TyAdt(def, _) if def.is_box() => {
            let typ = t.boxed_ty();
            if !typ.needs_drop(tcx, env) && type_is_sized(tcx, typ) {
                let layout = t.layout(tcx, ty::ParamEnv::empty(Reveal::All)).unwrap();
                // `Box<ZeroSizeType>` does not allocate.
                layout.size(&tcx.data_layout).bytes() != 0
            } else {
                true
            }
        }
        _ => true
    }
}

fn resolve_associated_item<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    trait_item: &ty::AssociatedItem,
    trait_id: DefId,
    rcvr_substs: &'tcx Substs<'tcx>
) -> ty::Instance<'tcx> {
    let def_id = trait_item.def_id;
    debug!("resolve_associated_item(trait_item={:?}, \
                                    trait_id={:?}, \
                                    rcvr_substs={:?})",
           def_id, trait_id, rcvr_substs);

    let trait_ref = ty::TraitRef::from_method(tcx, trait_id, rcvr_substs);
    let vtbl = fulfill_obligation(tcx, DUMMY_SP, ty::Binder(trait_ref));

    // Now that we know which impl is being used, we can dispatch to
    // the actual function:
    match vtbl {
        ::rustc::traits::VtableImpl(impl_data) => {
            let (def_id, substs) = ::rustc::traits::find_associated_item(
                tcx, trait_item, rcvr_substs, &impl_data);
            let substs = tcx.erase_regions(&substs);
            ty::Instance::new(def_id, substs)
        }
        ::rustc::traits::VtableClosure(closure_data) => {
            let trait_closure_kind = tcx.lang_items.fn_trait_kind(trait_id).unwrap();
            resolve_closure(tcx, closure_data.closure_def_id, closure_data.substs,
                            trait_closure_kind)
        }
        ::rustc::traits::VtableFnPointer(ref data) => {
            ty::Instance {
                def: ty::InstanceDef::FnPtrShim(trait_item.def_id, data.fn_ty),
                substs: rcvr_substs
            }
        }
        ::rustc::traits::VtableObject(ref data) => {
            let index = tcx.get_vtable_index_of_object_method(data, def_id);
            ty::Instance {
                def: ty::InstanceDef::Virtual(def_id, index),
                substs: rcvr_substs
            }
        }
        _ => {
            bug!("static call to invalid vtable: {:?}", vtbl)
        }
    }
}

pub fn def_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                        def_id: DefId,
                        substs: &'tcx Substs<'tcx>)
                        -> Ty<'tcx>
{
    let ty = tcx.type_of(def_id);
    apply_param_substs(tcx, substs, &ty)
}

/// Monomorphizes a type from the AST by first applying the in-scope
/// substitutions and then normalizing any associated types.
pub fn apply_param_substs<'a, 'tcx, T>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       param_substs: &Substs<'tcx>,
                                       value: &T)
                                       -> T
    where T: ::rustc::infer::TransNormalize<'tcx>
{
    debug!("apply_param_substs(param_substs={:?}, value={:?})", param_substs, value);
    let substituted = value.subst(tcx, param_substs);
    let substituted = tcx.erase_regions(&substituted);
    AssociatedTypeNormalizer{ tcx }.fold(&substituted)
}


struct AssociatedTypeNormalizer<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl<'a, 'tcx> AssociatedTypeNormalizer<'a, 'tcx> {
    fn fold<T: TypeFoldable<'tcx>>(&mut self, value: &T) -> T {
        if !value.has_projection_types() {
            value.clone()
        } else {
            value.fold_with(self)
        }
    }
}

impl<'a, 'tcx> ::rustc::ty::fold::TypeFolder<'tcx, 'tcx> for AssociatedTypeNormalizer<'a, 'tcx> {
    fn tcx<'c>(&'c self) -> TyCtxt<'c, 'tcx, 'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !ty.has_projection_types() {
            ty
        } else {
            self.tcx.normalize_associated_type(&ty)
        }
    }
}

fn type_is_sized<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> bool {
    // generics are weird, don't run this function on a generic
    assert!(!ty.needs_subst());
    ty.is_sized(tcx, ty::ParamEnv::empty(Reveal::All), DUMMY_SP)
}

/// Attempts to resolve an obligation. The result is a shallow vtable resolution -- meaning that we
/// do not (necessarily) resolve all nested obligations on the impl. Note that type check should
/// guarantee to us that all nested obligations *could be* resolved if we wanted to.
fn fulfill_obligation<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                span: Span,
                                trait_ref: ty::PolyTraitRef<'tcx>)
                                -> traits::Vtable<'tcx, ()>
{
    // Remove any references to regions; this helps improve caching.
    let trait_ref = tcx.erase_regions(&trait_ref);

    debug!("trans::fulfill_obligation(trait_ref={:?}, def_id={:?})",
            trait_ref, trait_ref.def_id());

    // Do the initial selection for the obligation. This yields the
    // shallow result we are looking for -- that is, what specific impl.
    tcx.infer_ctxt().enter(|infcx| {
        let mut selcx = traits::SelectionContext::new(&infcx);

        let obligation_cause = traits::ObligationCause::misc(span,
                                                            ast::DUMMY_NODE_ID);
        let obligation = traits::Obligation::new(obligation_cause,
                                                 ty::ParamEnv::empty(Reveal::All),
                                                 trait_ref.to_poly_trait_predicate());

        let selection = match selcx.select(&obligation) {
            Ok(Some(selection)) => selection,
            Ok(None) => {
                // Ambiguity can happen when monomorphizing during trans
                // expands to some humongo type that never occurred
                // statically -- this humongo type can then overflow,
                // leading to an ambiguous result. So report this as an
                // overflow bug, since I believe this is the only case
                // where ambiguity can result.
                debug!("Encountered ambiguity selecting `{:?}` during trans, \
                        presuming due to overflow",
                        trait_ref);
                tcx.sess.span_fatal(span,
                    "reached the recursion limit during monomorphization \
                        (selection ambiguity)");
            }
            Err(e) => {
                span_bug!(span, "Encountered error `{:?}` selecting `{:?}` during trans",
                            e, trait_ref)
            }
        };

        debug!("fulfill_obligation: selection={:?}", selection);

        // Currently, we use a fulfillment context to completely resolve
        // all nested obligations. This is because they can inform the
        // inference of the impl's type parameters.
        let mut fulfill_cx = traits::FulfillmentContext::new();
        let vtable = selection.map(|predicate| {
            debug!("fulfill_obligation: register_predicate_obligation {:?}", predicate);
            fulfill_cx.register_predicate_obligation(&infcx, predicate);
        });
        let vtable = infcx.drain_fulfillment_cx_or_panic(span, &mut fulfill_cx, &vtable);

        debug!("Cache miss: {:?} => {:?}", trait_ref, vtable);
        vtable
    })
}

pub fn resolve_drop_in_place<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    ty: Ty<'tcx>,
) -> ty::Instance<'tcx>
{
    let def_id = tcx.require_lang_item(::rustc::middle::lang_items::DropInPlaceFnLangItem);
    let substs = tcx.intern_substs(&[Kind::from(ty)]);
    resolve(tcx, def_id, substs)
}
