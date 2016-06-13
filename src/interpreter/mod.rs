use rustc::middle::const_val;
use rustc::hir::def_id::DefId;
use rustc::mir::mir_map::MirMap;
use rustc::mir::repr as mir;
use rustc::traits::{self, ProjectionMode};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::layout::{self, Layout, Size};
use rustc::ty::subst::{self, Subst, Substs};
use rustc::ty::{self, Ty, TyCtxt, BareFnTy};
use rustc::util::nodemap::DefIdMap;
use rustc_data_structures::indexed_vec::Idx;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use std::{iter, mem};
use syntax::ast;
use syntax::attr;
use syntax::codemap::{self, DUMMY_SP, Span};

use error::{EvalError, EvalResult};
use memory::{Memory, Pointer};
use primval::{self, PrimVal};

use std::collections::HashMap;

mod stepper;

pub fn step<'ecx, 'a: 'ecx, 'tcx: 'a>(ecx: &'ecx mut EvalContext<'a, 'tcx>) -> EvalResult<bool> {
    stepper::Stepper::new(ecx).step()
}

pub struct EvalContext<'a, 'tcx: 'a> {
    /// The results of the type checker, from rustc.
    tcx: TyCtxt<'a, 'tcx, 'tcx>,

    /// A mapping from NodeIds to Mir, from rustc. Only contains MIR for crate-local items.
    mir_map: &'a MirMap<'tcx>,

    /// A local cache from DefIds to Mir for non-crate-local items.
    mir_cache: RefCell<DefIdMap<Rc<mir::Mir<'tcx>>>>,

    /// The virtual memory system.
    memory: Memory<'tcx>,

    /// Precomputed statics, constants and promoteds
    statics: HashMap<ConstantId<'tcx>, Pointer>,

    /// The virtual call stack.
    stack: Vec<Frame<'a, 'tcx>>,
}

/// A stack frame.
pub struct Frame<'a, 'tcx: 'a> {
    /// The def_id of the current function
    pub def_id: DefId,

    /// The span of the call site
    pub span: codemap::Span,

    /// type substitutions for the current function invocation
    pub substs: &'tcx Substs<'tcx>,

    /// The MIR for the function called on this frame.
    pub mir: CachedMir<'a, 'tcx>,

    /// The block that is currently executed (or will be executed after the above call stacks return)
    pub next_block: mir::BasicBlock,

    /// A pointer for writing the return value of the current call if it's not a diverging call.
    pub return_ptr: Option<Pointer>,

    /// The list of locals for the current function, stored in order as
    /// `[arguments..., variables..., temporaries...]`. The variables begin at `self.var_offset`
    /// and the temporaries at `self.temp_offset`.
    pub locals: Vec<Pointer>,

    /// The offset of the first variable in `self.locals`.
    pub var_offset: usize,

    /// The offset of the first temporary in `self.locals`.
    pub temp_offset: usize,

    /// The index of the currently evaluated statment
    pub stmt: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Lvalue {
    ptr: Pointer,
    extra: LvalueExtra,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum LvalueExtra {
    None,
    Length(u64),
    // TODO(solson): Vtable(memory::AllocId),
    DowncastVariant(usize),
}

#[derive(Clone)]
pub enum CachedMir<'mir, 'tcx: 'mir> {
    Ref(&'mir mir::Mir<'tcx>),
    Owned(Rc<mir::Mir<'tcx>>)
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
/// Uniquely identifies a specific constant or static
struct ConstantId<'tcx> {
    /// the def id of the constant/static or in case of promoteds, the def id of the function they belong to
    def_id: DefId,
    /// In case of statics and constants this is `Substs::empty()`, so only promoteds and associated
    /// constants actually have something useful here. We could special case statics and constants,
    /// but that would only require more branching when working with constants, and not bring any
    /// real benefits.
    substs: &'tcx Substs<'tcx>,
    kind: ConstantKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum ConstantKind {
    Promoted(mir::Promoted),
    /// Statics, constants and associated constants
    Global,
}

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>, mir_map: &'a MirMap<'tcx>) -> Self {
        EvalContext {
            tcx: tcx,
            mir_map: mir_map,
            mir_cache: RefCell::new(DefIdMap()),
            memory: Memory::new(tcx.sess
                                   .target
                                   .uint_type
                                   .bit_width()
                                   .expect("Session::target::uint_type was usize")/8),
            statics: HashMap::new(),
            stack: Vec::new(),
        }
    }

    pub fn alloc_ret_ptr(&mut self, output_ty: ty::FnOutput<'tcx>, substs: &'tcx Substs<'tcx>) -> Option<Pointer> {
        match output_ty {
            ty::FnConverging(ty) => {
                let size = self.type_size_with_substs(ty, substs);
                Some(self.memory.allocate(size))
            }
            ty::FnDiverging => None,
        }
    }

    pub fn memory(&self) -> &Memory {
        &self.memory
    }

    pub fn stack(&self) -> &[Frame] {
        &self.stack
    }

    // TODO(solson): Try making const_to_primval instead.
    fn const_to_ptr(&mut self, const_val: &const_val::ConstVal) -> EvalResult<Pointer> {
        use rustc::middle::const_val::ConstVal::*;
        match *const_val {
            Float(_f) => unimplemented!(),
            Integral(int) => {
                // TODO(solson): Check int constant type.
                let ptr = self.memory.allocate(8);
                self.memory.write_uint(ptr, int.to_u64_unchecked(), 8)?;
                Ok(ptr)
            }
            Str(ref s) => {
                let psize = self.memory.pointer_size;
                let static_ptr = self.memory.allocate(s.len());
                let ptr = self.memory.allocate(psize * 2);
                self.memory.write_bytes(static_ptr, s.as_bytes())?;
                self.memory.write_ptr(ptr, static_ptr)?;
                self.memory.write_usize(ptr.offset(psize as isize), s.len() as u64)?;
                Ok(ptr)
            }
            ByteStr(ref bs) => {
                let psize = self.memory.pointer_size;
                let static_ptr = self.memory.allocate(bs.len());
                let ptr = self.memory.allocate(psize);
                self.memory.write_bytes(static_ptr, bs)?;
                self.memory.write_ptr(ptr, static_ptr)?;
                Ok(ptr)
            }
            Bool(b) => {
                let ptr = self.memory.allocate(1);
                self.memory.write_bool(ptr, b)?;
                Ok(ptr)
            }
            Char(_c)          => unimplemented!(),
            Struct(_node_id)  => unimplemented!(),
            Tuple(_node_id)   => unimplemented!(),
            Function(_def_id) => unimplemented!(),
            Array(_, _)       => unimplemented!(),
            Repeat(_, _)      => unimplemented!(),
            Dummy             => unimplemented!(),
        }
    }

    fn type_needs_drop(&self, ty: Ty<'tcx>) -> bool {
        self.tcx.type_needs_drop_given_env(ty, &self.tcx.empty_parameter_environment())
    }

    fn type_is_sized(&self, ty: Ty<'tcx>) -> bool {
        ty.is_sized(self.tcx, &self.tcx.empty_parameter_environment(), DUMMY_SP)
    }

    fn fulfill_obligation(&self, trait_ref: ty::PolyTraitRef<'tcx>) -> traits::Vtable<'tcx, ()> {
        // Do the initial selection for the obligation. This yields the shallow result we are
        // looking for -- that is, what specific impl.
        self.tcx.normalizing_infer_ctxt(ProjectionMode::Any).enter(|infcx| {
            let mut selcx = traits::SelectionContext::new(&infcx);

            let obligation = traits::Obligation::new(
                traits::ObligationCause::misc(DUMMY_SP, ast::DUMMY_NODE_ID),
                trait_ref.to_poly_trait_predicate(),
            );
            let selection = selcx.select(&obligation).unwrap().unwrap();

            // Currently, we use a fulfillment context to completely resolve all nested obligations.
            // This is because they can inform the inference of the impl's type parameters.
            let mut fulfill_cx = traits::FulfillmentContext::new();
            let vtable = selection.map(|predicate| {
                fulfill_cx.register_predicate_obligation(&infcx, predicate);
            });
            infcx.drain_fulfillment_cx_or_panic(DUMMY_SP, &mut fulfill_cx, &vtable)
        })
    }

    /// Trait method, which has to be resolved to an impl method.
    pub fn trait_method(
        &self,
        def_id: DefId,
        substs: &'tcx Substs<'tcx>
    ) -> (DefId, &'tcx Substs<'tcx>) {
        let method_item = self.tcx.impl_or_trait_item(def_id);
        let trait_id = method_item.container().id();
        let trait_ref = ty::Binder(substs.to_trait_ref(self.tcx, trait_id));
        match self.fulfill_obligation(trait_ref) {
            traits::VtableImpl(vtable_impl) => {
                let impl_did = vtable_impl.impl_def_id;
                let mname = self.tcx.item_name(def_id);
                // Create a concatenated set of substitutions which includes those from the impl
                // and those from the method:
                let impl_substs = vtable_impl.substs.with_method_from(substs);
                let substs = self.tcx.mk_substs(impl_substs);
                let mth = get_impl_method(self.tcx, impl_did, substs, mname);

                (mth.method.def_id, mth.substs)
            }

            traits::VtableClosure(vtable_closure) =>
                (vtable_closure.closure_def_id, vtable_closure.substs.func_substs),

            traits::VtableFnPointer(_fn_ty) => {
                let _trait_closure_kind = self.tcx.lang_items.fn_trait_kind(trait_id).unwrap();
                unimplemented!()
                // let llfn = trans_fn_pointer_shim(ccx, trait_closure_kind, fn_ty);

                // let method_ty = def_ty(tcx, def_id, substs);
                // let fn_ptr_ty = match method_ty.sty {
                //     ty::TyFnDef(_, _, fty) => tcx.mk_ty(ty::TyFnPtr(fty)),
                //     _ => unreachable!("expected fn item type, found {}",
                //                       method_ty)
                // };
                // Callee::ptr(immediate_rvalue(llfn, fn_ptr_ty))
            }

            traits::VtableObject(ref _data) => {
                unimplemented!()
                // Callee {
                //     data: Virtual(traits::get_vtable_index_of_object_method(
                //                   tcx, data, def_id)),
                //                   ty: def_ty(tcx, def_id, substs)
                // }
            }
            vtable => unreachable!("resolved vtable bad vtable {:?} in trans", vtable),
        }
    }

    fn load_mir(&self, def_id: DefId) -> CachedMir<'a, 'tcx> {
        use rustc_trans::back::symbol_names::def_id_to_string;
        match self.tcx.map.as_local_node_id(def_id) {
            Some(node_id) => CachedMir::Ref(self.mir_map.map.get(&node_id).unwrap()),
            None => {
                let mut mir_cache = self.mir_cache.borrow_mut();
                if let Some(mir) = mir_cache.get(&def_id) {
                    return CachedMir::Owned(mir.clone());
                }

                let cs = &self.tcx.sess.cstore;
                let mir = cs.maybe_get_item_mir(self.tcx, def_id).unwrap_or_else(|| {
                    panic!("no mir for `{}`", def_id_to_string(self.tcx, def_id));
                });
                let cached = Rc::new(mir);
                mir_cache.insert(def_id, cached.clone());
                CachedMir::Owned(cached)
            }
        }
    }

    fn monomorphize(&self, ty: Ty<'tcx>, substs: &'tcx Substs<'tcx>) -> Ty<'tcx> {
        let substituted = ty.subst(self.tcx, substs);
        self.tcx.normalize_associated_type(&substituted)
    }

    fn type_size(&self, ty: Ty<'tcx>) -> usize {
        self.type_size_with_substs(ty, self.substs())
    }

    fn type_size_with_substs(&self, ty: Ty<'tcx>, substs: &'tcx Substs<'tcx>) -> usize {
        self.type_layout_with_substs(ty, substs).size(&self.tcx.data_layout).bytes() as usize
    }

    fn type_layout(&self, ty: Ty<'tcx>) -> &'tcx Layout {
        self.type_layout_with_substs(ty, self.substs())
    }

    fn type_layout_with_substs(&self, ty: Ty<'tcx>, substs: &'tcx Substs<'tcx>) -> &'tcx Layout {
        // TODO(solson): Is this inefficient? Needs investigation.
        let ty = self.monomorphize(ty, substs);

        self.tcx.normalizing_infer_ctxt(ProjectionMode::Any).enter(|infcx| {
            // TODO(solson): Report this error properly.
            ty.layout(&infcx).unwrap()
        })
    }

    pub fn push_stack_frame(&mut self, def_id: DefId, span: codemap::Span, mir: CachedMir<'a, 'tcx>, substs: &'tcx Substs<'tcx>,
        return_ptr: Option<Pointer>)
    {
        let arg_tys = mir.arg_decls.iter().map(|a| a.ty);
        let var_tys = mir.var_decls.iter().map(|v| v.ty);
        let temp_tys = mir.temp_decls.iter().map(|t| t.ty);

        let num_args = mir.arg_decls.len();
        let num_vars = mir.var_decls.len();

        ::log_settings::settings().indentation += 1;

        let locals: Vec<Pointer> = arg_tys.chain(var_tys).chain(temp_tys).map(|ty| {
            let size = self.type_size_with_substs(ty, substs);
            self.memory.allocate(size)
        }).collect();

        self.stack.push(Frame {
            mir: mir.clone(),
            next_block: mir::START_BLOCK,
            return_ptr: return_ptr,
            locals: locals,
            var_offset: num_args,
            temp_offset: num_args + num_vars,
            span: span,
            def_id: def_id,
            substs: substs,
            stmt: 0,
        });
    }

    fn pop_stack_frame(&mut self) {
        ::log_settings::settings().indentation -= 1;
        let _frame = self.stack.pop().expect("tried to pop a stack frame, but there were none");
        // TODO(solson): Deallocate local variables.
    }

    fn eval_terminator(&mut self, terminator: &mir::Terminator<'tcx>)
            -> EvalResult<()> {
        use rustc::mir::repr::TerminatorKind::*;
        match terminator.kind {
            Return => self.pop_stack_frame(),

            Goto { target } => {
                self.frame_mut().next_block = target;
            },

            If { ref cond, targets: (then_target, else_target) } => {
                let cond_ptr = self.eval_operand(cond)?;
                let cond_val = self.memory.read_bool(cond_ptr)?;
                self.frame_mut().next_block = if cond_val { then_target } else { else_target };
            }

            SwitchInt { ref discr, ref values, ref targets, .. } => {
                let discr_ptr = self.eval_lvalue(discr)?.to_ptr();
                let discr_size = self
                    .type_layout(self.lvalue_ty(discr))
                    .size(&self.tcx.data_layout)
                    .bytes() as usize;
                let discr_val = self.memory.read_uint(discr_ptr, discr_size)?;

                // Branch to the `otherwise` case by default, if no match is found.
                let mut target_block = targets[targets.len() - 1];

                for (index, val_const) in values.iter().enumerate() {
                    let ptr = self.const_to_ptr(val_const)?;
                    let val = self.memory.read_uint(ptr, discr_size)?;
                    if discr_val == val {
                        target_block = targets[index];
                        break;
                    }
                }

                self.frame_mut().next_block = target_block;
            }

            Switch { ref discr, ref targets, adt_def } => {
                let adt_ptr = self.eval_lvalue(discr)?.to_ptr();
                let adt_ty = self.lvalue_ty(discr);
                let discr_val = self.read_discriminant_value(adt_ptr, adt_ty)?;
                let matching = adt_def.variants.iter()
                    .position(|v| discr_val == v.disr_val.to_u64_unchecked());

                match matching {
                    Some(i) => {
                        self.frame_mut().next_block = targets[i];
                    },
                    None => return Err(EvalError::InvalidDiscriminant),
                }
            }

            Call { ref func, ref args, ref destination, .. } => {
                let mut return_ptr = None;
                if let Some((ref lv, target)) = *destination {
                    self.frame_mut().next_block = target;
                    return_ptr = Some(self.eval_lvalue(lv)?.to_ptr());
                }

                let func_ty = self.operand_ty(func);
                match func_ty.sty {
                    ty::TyFnPtr(bare_fn_ty) => {
                        let ptr = self.eval_operand(func)?;
                        assert_eq!(ptr.offset, 0);
                        let fn_ptr = self.memory.read_ptr(ptr)?;
                        let (def_id, substs) = self.memory.get_fn(fn_ptr.alloc_id)?;
                        self.eval_fn_call(def_id, substs, bare_fn_ty, return_ptr, args,
                                          terminator.source_info.span)?
                    },
                    ty::TyFnDef(def_id, substs, fn_ty) => {
                        self.eval_fn_call(def_id, substs, fn_ty, return_ptr, args,
                                          terminator.source_info.span)?
                    }

                    _ => return Err(EvalError::Unimplemented(format!("can't handle callee of type {:?}", func_ty))),
                }
            }

            Drop { ref location, target, .. } => {
                let ptr = self.eval_lvalue(location)?.to_ptr();
                let ty = self.lvalue_ty(location);
                self.drop(ptr, ty)?;
                self.frame_mut().next_block = target;
            }

            Assert { ref cond, expected, ref msg, target, cleanup } => {
                let actual_ptr = self.eval_operand(cond)?;
                let actual = self.memory.read_bool(actual_ptr)?;
                if actual == expected {
                    self.frame_mut().next_block = target;
                } else {
                    panic!("unimplemented: jump to {:?} and print {:?}", cleanup, msg);
                }
            }

            DropAndReplace { .. } => unimplemented!(),
            Resume => unimplemented!(),
            Unreachable => unimplemented!(),
        }

        Ok(())
    }

    pub fn eval_fn_call(
        &mut self,
        def_id: DefId,
        substs: &'tcx Substs<'tcx>,
        fn_ty: &'tcx BareFnTy,
        return_ptr: Option<Pointer>,
        args: &[mir::Operand<'tcx>],
        span: Span,
    ) -> EvalResult<()> {
        use syntax::abi::Abi;
        match fn_ty.abi {
            Abi::RustIntrinsic => {
                let name = self.tcx.item_name(def_id).as_str();
                match fn_ty.sig.0.output {
                    ty::FnConverging(ty) => {
                        let size = self.type_size(ty);
                        let ret = return_ptr.unwrap();
                        self.call_intrinsic(&name, substs, args, ret, size)
                    }
                    ty::FnDiverging => unimplemented!(),
                }
            }

            Abi::C => {
                match fn_ty.sig.0.output {
                    ty::FnConverging(ty) => {
                        let size = self.type_size(ty);
                        self.call_c_abi(def_id, args, return_ptr.unwrap(), size)
                    }
                    ty::FnDiverging => unimplemented!(),
                }
            }

            Abi::Rust | Abi::RustCall => {
                // TODO(solson): Adjust the first argument when calling a Fn or
                // FnMut closure via FnOnce::call_once.

                // Only trait methods can have a Self parameter.
                let (resolved_def_id, resolved_substs) = if substs.self_ty().is_some() {
                    self.trait_method(def_id, substs)
                } else {
                    (def_id, substs)
                };

                let mut arg_srcs = Vec::new();
                for arg in args {
                    let src = self.eval_operand(arg)?;
                    let src_ty = self.operand_ty(arg);
                    arg_srcs.push((src, src_ty));
                }

                if fn_ty.abi == Abi::RustCall && !args.is_empty() {
                    arg_srcs.pop();
                    let last_arg = args.last().unwrap();
                    let last = self.eval_operand(last_arg)?;
                    let last_ty = self.operand_ty(last_arg);
                    let last_layout = self.type_layout(last_ty);
                    match (&last_ty.sty, last_layout) {
                        (&ty::TyTuple(fields),
                         &Layout::Univariant { ref variant, .. }) => {
                            let offsets = iter::once(0)
                                .chain(variant.offset_after_field.iter()
                                    .map(|s| s.bytes()));
                            for (offset, ty) in offsets.zip(fields) {
                                let src = last.offset(offset as isize);
                                arg_srcs.push((src, ty));
                            }
                        }
                        ty => panic!("expected tuple as last argument in function with 'rust-call' ABI, got {:?}", ty),
                    }
                }

                let mir = self.load_mir(resolved_def_id);
                self.push_stack_frame(def_id, span, mir, resolved_substs, return_ptr);

                for (i, (src, src_ty)) in arg_srcs.into_iter().enumerate() {
                    let dest = self.frame().locals[i];
                    self.move_(src, dest, src_ty)?;
                }

                Ok(())
            }

            abi => Err(EvalError::Unimplemented(format!("can't handle function with {:?} ABI", abi))),
        }
    }

    fn drop(&mut self, ptr: Pointer, ty: Ty<'tcx>) -> EvalResult<()> {
        if !self.type_needs_drop(ty) {
            debug!("no need to drop {:?}", ty);
            return Ok(());
        }
        trace!("-need to drop {:?}", ty);

        // TODO(solson): Call user-defined Drop::drop impls.

        match ty.sty {
            ty::TyBox(contents_ty) => {
                match self.memory.read_ptr(ptr) {
                    Ok(contents_ptr) => {
                        self.drop(contents_ptr, contents_ty)?;
                        trace!("-deallocating box");
                        self.memory.deallocate(contents_ptr)?;
                    }
                    Err(EvalError::ReadBytesAsPointer) => {
                        let size = self.memory.pointer_size;
                        let possible_drop_fill = self.memory.read_bytes(ptr, size)?;
                        if possible_drop_fill.iter().all(|&b| b == mem::POST_DROP_U8) {
                            return Ok(());
                        } else {
                            return Err(EvalError::ReadBytesAsPointer);
                        }
                    }
                    Err(e) => return Err(e),
                }
            }

            // TODO(solson): Implement drop for other relevant types (e.g. aggregates).
            _ => {}
        }

        // Filling drop.
        // FIXME(solson): Trait objects (with no static size) probably get filled, too.
        let size = self.type_size(ty);
        self.memory.drop_fill(ptr, size)?;

        Ok(())
    }

    fn read_discriminant_value(&self, adt_ptr: Pointer, adt_ty: Ty<'tcx>) -> EvalResult<u64> {
        use rustc::ty::layout::Layout::*;
        let adt_layout = self.type_layout(adt_ty);

        let discr_val = match *adt_layout {
            General { discr, .. } | CEnum { discr, .. } => {
                let discr_size = discr.size().bytes();
                self.memory.read_uint(adt_ptr, discr_size as usize)?
            }

            RawNullablePointer { nndiscr, .. } => {
                self.read_nonnull_discriminant_value(adt_ptr, nndiscr)?
            }

            StructWrappedNullablePointer { nndiscr, ref discrfield, .. } => {
                let offset = self.nonnull_offset(adt_ty, nndiscr, discrfield)?;
                let nonnull = adt_ptr.offset(offset.bytes() as isize);
                self.read_nonnull_discriminant_value(nonnull, nndiscr)?
            }

            // The discriminant_value intrinsic returns 0 for non-sum types.
            Array { .. } | FatPointer { .. } | Scalar { .. } | Univariant { .. } |
            Vector { .. } => 0,
        };

        Ok(discr_val)
    }

    fn read_nonnull_discriminant_value(&self, ptr: Pointer, nndiscr: u64) -> EvalResult<u64> {
        let not_null = match self.memory.read_usize(ptr) {
            Ok(0) => false,
            Ok(_) | Err(EvalError::ReadPointerAsBytes) => true,
            Err(e) => return Err(e),
        };
        assert!(nndiscr == 0 || nndiscr == 1);
        Ok(if not_null { nndiscr } else { 1 - nndiscr })
    }

    fn call_intrinsic(
        &mut self,
        name: &str,
        substs: &'tcx Substs<'tcx>,
        args: &[mir::Operand<'tcx>],
        dest: Pointer,
        dest_size: usize
    ) -> EvalResult<()> {
        let args_res: EvalResult<Vec<Pointer>> = args.iter()
            .map(|arg| self.eval_operand(arg))
            .collect();
        let args = args_res?;

        match name {
            // FIXME(solson): Handle different integer types correctly.
            "add_with_overflow" => {
                let ty = *substs.types.get(subst::FnSpace, 0);
                let size = self.type_size(ty);
                let left = self.memory.read_int(args[0], size)?;
                let right = self.memory.read_int(args[1], size)?;
                let (n, overflowed) = unsafe {
                    ::std::intrinsics::add_with_overflow::<i64>(left, right)
                };
                self.memory.write_int(dest, n, size)?;
                self.memory.write_bool(dest.offset(size as isize), overflowed)?;
            }

            "assume" => {}

            "copy_nonoverlapping" => {
                let elem_ty = *substs.types.get(subst::FnSpace, 0);
                let elem_size = self.type_size(elem_ty);
                let src = self.memory.read_ptr(args[0])?;
                let dest = self.memory.read_ptr(args[1])?;
                let count = self.memory.read_isize(args[2])?;
                self.memory.copy(src, dest, count as usize * elem_size)?;
            }

            "discriminant_value" => {
                let ty = *substs.types.get(subst::FnSpace, 0);
                let adt_ptr = self.memory.read_ptr(args[0])?;
                let discr_val = self.read_discriminant_value(adt_ptr, ty)?;
                self.memory.write_uint(dest, discr_val, dest_size)?;
            }

            "forget" => {
                let arg_ty = *substs.types.get(subst::FnSpace, 0);
                let arg_size = self.type_size(arg_ty);
                self.memory.drop_fill(args[0], arg_size)?;
            }

            "init" => self.memory.write_repeat(dest, 0, dest_size)?,

            "min_align_of" => {
                self.memory.write_int(dest, 1, dest_size)?;
            }

            "move_val_init" => {
                let ty = *substs.types.get(subst::FnSpace, 0);
                let ptr = self.memory.read_ptr(args[0])?;
                self.move_(args[1], ptr, ty)?;
            }

            // FIXME(solson): Handle different integer types correctly.
            "mul_with_overflow" => {
                let ty = *substs.types.get(subst::FnSpace, 0);
                let size = self.type_size(ty);
                let left = self.memory.read_int(args[0], size)?;
                let right = self.memory.read_int(args[1], size)?;
                let (n, overflowed) = unsafe {
                    ::std::intrinsics::mul_with_overflow::<i64>(left, right)
                };
                self.memory.write_int(dest, n, size)?;
                self.memory.write_bool(dest.offset(size as isize), overflowed)?;
            }

            "offset" => {
                let pointee_ty = *substs.types.get(subst::FnSpace, 0);
                let pointee_size = self.type_size(pointee_ty) as isize;
                let ptr_arg = args[0];
                let offset = self.memory.read_isize(args[1])?;

                match self.memory.read_ptr(ptr_arg) {
                    Ok(ptr) => {
                        let result_ptr = ptr.offset(offset as isize * pointee_size);
                        self.memory.write_ptr(dest, result_ptr)?;
                    }
                    Err(EvalError::ReadBytesAsPointer) => {
                        let addr = self.memory.read_isize(ptr_arg)?;
                        let result_addr = addr + offset * pointee_size as i64;
                        self.memory.write_isize(dest, result_addr)?;
                    }
                    Err(e) => return Err(e),
                }
            }

            // FIXME(solson): Handle different integer types correctly. Use primvals?
            "overflowing_sub" => {
                let ty = *substs.types.get(subst::FnSpace, 0);
                let size = self.type_size(ty);
                let left = self.memory.read_int(args[0], size)?;
                let right = self.memory.read_int(args[1], size)?;
                let n = left.wrapping_sub(right);
                self.memory.write_int(dest, n, size)?;
            }

            "size_of" => {
                let ty = *substs.types.get(subst::FnSpace, 0);
                let size = self.type_size(ty) as u64;
                self.memory.write_uint(dest, size, dest_size)?;
            }

            "size_of_val" => {
                let ty = *substs.types.get(subst::FnSpace, 0);
                if self.type_is_sized(ty) {
                    let size = self.type_size(ty) as u64;
                    self.memory.write_uint(dest, size, dest_size)?;
                } else {
                    match ty.sty {
                        ty::TySlice(_) | ty::TyStr => {
                            let elem_ty = ty.sequence_element_type(self.tcx);
                            let elem_size = self.type_size(elem_ty) as u64;
                            let ptr_size = self.memory.pointer_size as isize;
                            let n = self.memory.read_usize(args[0].offset(ptr_size))?;
                            self.memory.write_uint(dest, n * elem_size, dest_size)?;
                        }

                        _ => return Err(EvalError::Unimplemented(format!("unimplemented: size_of_val::<{:?}>", ty))),
                    }
                }
            }

            "transmute" => {
                let ty = *substs.types.get(subst::FnSpace, 0);
                self.move_(args[0], dest, ty)?;
            }
            "uninit" => self.memory.mark_definedness(dest, dest_size, false)?,

            name => return Err(EvalError::Unimplemented(format!("unimplemented intrinsic: {}", name))),
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        Ok(())
    }

    fn call_c_abi(
        &mut self,
        def_id: DefId,
        args: &[mir::Operand<'tcx>],
        dest: Pointer,
        dest_size: usize,
    ) -> EvalResult<()> {
        let name = self.tcx.item_name(def_id);
        let attrs = self.tcx.get_attrs(def_id);
        let link_name = match attr::first_attr_value_str_by_name(&attrs, "link_name") {
            Some(ln) => ln.clone(),
            None => name.as_str(),
        };

        let args_res: EvalResult<Vec<Pointer>> = args.iter()
            .map(|arg| self.eval_operand(arg))
            .collect();
        let args = args_res?;

        match &link_name[..] {
            "__rust_allocate" => {
                let size = self.memory.read_usize(args[0])?;
                let ptr = self.memory.allocate(size as usize);
                self.memory.write_ptr(dest, ptr)?;
            }

            "__rust_reallocate" => {
                let ptr = self.memory.read_ptr(args[0])?;
                let size = self.memory.read_usize(args[2])?;
                self.memory.reallocate(ptr, size as usize)?;
                self.memory.write_ptr(dest, ptr)?;
            }

            "memcmp" => {
                let left = self.memory.read_ptr(args[0])?;
                let right = self.memory.read_ptr(args[1])?;
                let n = self.memory.read_usize(args[2])? as usize;

                let result = {
                    let left_bytes = self.memory.read_bytes(left, n)?;
                    let right_bytes = self.memory.read_bytes(right, n)?;

                    use std::cmp::Ordering::*;
                    match left_bytes.cmp(right_bytes) {
                        Less => -1,
                        Equal => 0,
                        Greater => 1,
                    }
                };

                self.memory.write_int(dest, result, dest_size)?;
            }

            _ => {
                return Err(EvalError::Unimplemented(format!("can't call C ABI function: {}", link_name)));
            }
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        Ok(())
    }

    fn assign_fields<I: IntoIterator<Item = u64>>(
        &mut self,
        dest: Pointer,
        offsets: I,
        operands: &[mir::Operand<'tcx>],
    ) -> EvalResult<()> {
        for (offset, operand) in offsets.into_iter().zip(operands) {
            let src = self.eval_operand(operand)?;
            let src_ty = self.operand_ty(operand);
            let field_dest = dest.offset(offset as isize);
            self.move_(src, field_dest, src_ty)?;
        }
        Ok(())
    }

    fn eval_assignment(&mut self, lvalue: &mir::Lvalue<'tcx>, rvalue: &mir::Rvalue<'tcx>)
        -> EvalResult<()>
    {
        let dest = self.eval_lvalue(lvalue)?.to_ptr();
        let dest_ty = self.lvalue_ty(lvalue);
        let dest_layout = self.type_layout(dest_ty);

        use rustc::mir::repr::Rvalue::*;
        match *rvalue {
            Use(ref operand) => {
                let src = self.eval_operand(operand)?;
                self.move_(src, dest, dest_ty)?;
            }

            BinaryOp(bin_op, ref left, ref right) => {
                let left_ptr = self.eval_operand(left)?;
                let left_ty = self.operand_ty(left);
                let left_val = self.read_primval(left_ptr, left_ty)?;

                let right_ptr = self.eval_operand(right)?;
                let right_ty = self.operand_ty(right);
                let right_val = self.read_primval(right_ptr, right_ty)?;

                let val = primval::binary_op(bin_op, left_val, right_val)?;
                self.memory.write_primval(dest, val)?;
            }

            // FIXME(solson): Factor this out with BinaryOp.
            CheckedBinaryOp(bin_op, ref left, ref right) => {
                let left_ptr = self.eval_operand(left)?;
                let left_ty = self.operand_ty(left);
                let left_val = self.read_primval(left_ptr, left_ty)?;

                let right_ptr = self.eval_operand(right)?;
                let right_ty = self.operand_ty(right);
                let right_val = self.read_primval(right_ptr, right_ty)?;

                let val = primval::binary_op(bin_op, left_val, right_val)?;
                self.memory.write_primval(dest, val)?;

                // FIXME(solson): Find the result type size properly. Perhaps refactor out
                // Projection calculations so we can do the equivalent of `dest.1` here.
                let s = self.type_size(left_ty);
                self.memory.write_bool(dest.offset(s as isize), false)?;
            }

            UnaryOp(un_op, ref operand) => {
                let ptr = self.eval_operand(operand)?;
                let ty = self.operand_ty(operand);
                let val = self.read_primval(ptr, ty)?;
                self.memory.write_primval(dest, primval::unary_op(un_op, val)?)?;
            }

            Aggregate(ref kind, ref operands) => {
                use rustc::ty::layout::Layout::*;
                match *dest_layout {
                    Univariant { ref variant, .. } => {
                        let offsets = iter::once(0)
                            .chain(variant.offset_after_field.iter().map(|s| s.bytes()));
                        self.assign_fields(dest, offsets, operands)?;
                    }

                    Array { .. } => {
                        let elem_size = match dest_ty.sty {
                            ty::TyArray(elem_ty, _) => self.type_size(elem_ty) as u64,
                            _ => panic!("tried to assign {:?} to non-array type {:?}",
                                        kind, dest_ty),
                        };
                        let offsets = (0..).map(|i| i * elem_size);
                        self.assign_fields(dest, offsets, operands)?;
                    }

                    General { discr, ref variants, .. } => {
                        if let mir::AggregateKind::Adt(adt_def, variant, _) = *kind {
                            let discr_val = adt_def.variants[variant].disr_val.to_u64_unchecked();
                            let discr_size = discr.size().bytes() as usize;
                            self.memory.write_uint(dest, discr_val, discr_size)?;

                            let offsets = variants[variant].offset_after_field.iter()
                                .map(|s| s.bytes());
                            self.assign_fields(dest, offsets, operands)?;
                        } else {
                            panic!("tried to assign {:?} to Layout::General", kind);
                        }
                    }

                    RawNullablePointer { nndiscr, .. } => {
                        if let mir::AggregateKind::Adt(_, variant, _) = *kind {
                            if nndiscr == variant as u64 {
                                assert_eq!(operands.len(), 1);
                                let operand = &operands[0];
                                let src = self.eval_operand(operand)?;
                                let src_ty = self.operand_ty(operand);
                                self.move_(src, dest, src_ty)?;
                            } else {
                                assert_eq!(operands.len(), 0);
                                self.memory.write_isize(dest, 0)?;
                            }
                        } else {
                            panic!("tried to assign {:?} to Layout::RawNullablePointer", kind);
                        }
                    }

                    StructWrappedNullablePointer { nndiscr, ref nonnull, ref discrfield } => {
                        if let mir::AggregateKind::Adt(_, variant, _) = *kind {
                            if nndiscr == variant as u64 {
                                let offsets = iter::once(0)
                                    .chain(nonnull.offset_after_field.iter().map(|s| s.bytes()));
                                try!(self.assign_fields(dest, offsets, operands));
                            } else {
                                assert_eq!(operands.len(), 0);
                                let offset = self.nonnull_offset(dest_ty, nndiscr, discrfield)?;
                                let dest = dest.offset(offset.bytes() as isize);
                                try!(self.memory.write_isize(dest, 0));
                            }
                        } else {
                            panic!("tried to assign {:?} to Layout::RawNullablePointer", kind);
                        }
                    }

                    CEnum { discr, signed, .. } => {
                        assert_eq!(operands.len(), 0);
                        if let mir::AggregateKind::Adt(adt_def, variant, _) = *kind {
                            let val = adt_def.variants[variant].disr_val.to_u64_unchecked();
                            let size = discr.size().bytes() as usize;

                            if signed {
                                self.memory.write_int(dest, val as i64, size)?;
                            } else {
                                self.memory.write_uint(dest, val, size)?;
                            }
                        } else {
                            panic!("tried to assign {:?} to Layout::CEnum", kind);
                        }
                    }

                    _ => return Err(EvalError::Unimplemented(format!("can't handle destination layout {:?} when assigning {:?}", dest_layout, kind))),
                }
            }

            Repeat(ref operand, _) => {
                let (elem_size, length) = match dest_ty.sty {
                    ty::TyArray(elem_ty, n) => (self.type_size(elem_ty), n),
                    _ => panic!("tried to assign array-repeat to non-array type {:?}", dest_ty),
                };

                let src = self.eval_operand(operand)?;
                for i in 0..length {
                    let elem_dest = dest.offset((i * elem_size) as isize);
                    self.memory.copy(src, elem_dest, elem_size)?;
                }
            }

            Len(ref lvalue) => {
                let src = self.eval_lvalue(lvalue)?;
                let ty = self.lvalue_ty(lvalue);
                let len = match ty.sty {
                    ty::TyArray(_, n) => n as u64,
                    ty::TySlice(_) => if let LvalueExtra::Length(n) = src.extra {
                        n
                    } else {
                        panic!("Rvalue::Len of a slice given non-slice pointer: {:?}", src);
                    },
                    _ => panic!("Rvalue::Len expected array or slice, got {:?}", ty),
                };
                self.memory.write_usize(dest, len)?;
            }

            Ref(_, _, ref lvalue) => {
                let lv = self.eval_lvalue(lvalue)?;
                self.memory.write_ptr(dest, lv.ptr)?;
                match lv.extra {
                    LvalueExtra::None => {},
                    LvalueExtra::Length(len) => {
                        let len_ptr = dest.offset(self.memory.pointer_size as isize);
                        self.memory.write_usize(len_ptr, len)?;
                    }
                    LvalueExtra::DowncastVariant(..) =>
                        panic!("attempted to take a reference to an enum downcast lvalue"),
                }
            }

            Box(ty) => {
                let size = self.type_size(ty);
                let ptr = self.memory.allocate(size);
                self.memory.write_ptr(dest, ptr)?;
            }

            Cast(kind, ref operand, dest_ty) => {
                use rustc::mir::repr::CastKind::*;
                match kind {
                    Unsize => {
                        let src = self.eval_operand(operand)?;
                        let src_ty = self.operand_ty(operand);
                        self.move_(src, dest, src_ty)?;
                        let src_pointee_ty = pointee_type(src_ty).unwrap();
                        let dest_pointee_ty = pointee_type(dest_ty).unwrap();

                        match (&src_pointee_ty.sty, &dest_pointee_ty.sty) {
                            (&ty::TyArray(_, length), &ty::TySlice(_)) => {
                                let len_ptr = dest.offset(self.memory.pointer_size as isize);
                                self.memory.write_usize(len_ptr, length as u64)?;
                            }

                            _ => return Err(EvalError::Unimplemented(format!("can't handle cast: {:?}", rvalue))),
                        }
                    }

                    Misc => {
                        let src = self.eval_operand(operand)?;
                        let src_ty = self.operand_ty(operand);
                        // FIXME(solson): Wrong for almost everything.
                        warn!("misc cast from {:?} to {:?}", src_ty, dest_ty);
                        let dest_size = self.type_size(dest_ty);
                        let src_size = self.type_size(src_ty);

                        // Hack to support fat pointer -> thin pointer casts to keep tests for
                        // other things passing for now.
                        let is_fat_ptr_cast = pointee_type(src_ty).map(|ty| {
                            !self.type_is_sized(ty)
                        }).unwrap_or(false);

                        if dest_size == src_size || is_fat_ptr_cast {
                            self.memory.copy(src, dest, dest_size)?;
                        } else {
                            return Err(EvalError::Unimplemented(format!("can't handle cast: {:?}", rvalue)));
                        }
                    }

                    ReifyFnPointer => match self.operand_ty(operand).sty {
                        ty::TyFnDef(def_id, substs, _) => {
                            let fn_ptr = self.memory.create_fn_ptr(def_id, substs);
                            self.memory.write_ptr(dest, fn_ptr)?;
                        },
                        ref other => panic!("reify fn pointer on {:?}", other),
                    },

                    _ => return Err(EvalError::Unimplemented(format!("can't handle cast: {:?}", rvalue))),
                }
            }

            InlineAsm { .. } => unimplemented!(),
        }

        Ok(())
    }

    fn nonnull_offset(&self, ty: Ty<'tcx>, nndiscr: u64, discrfield: &[u32]) -> EvalResult<Size> {
        // Skip the constant 0 at the start meant for LLVM GEP.
        let mut path = discrfield.iter().skip(1).map(|&i| i as usize);

        // Handle the field index for the outer non-null variant.
        let inner_ty = match ty.sty {
            ty::TyEnum(adt_def, substs) => {
                let variant = &adt_def.variants[nndiscr as usize];
                let index = path.next().unwrap();
                let field = &variant.fields[index];
                field.ty(self.tcx, substs)
            }
            _ => panic!(
                "non-enum for StructWrappedNullablePointer: {}",
                ty,
            ),
        };

        self.field_path_offset(inner_ty, path)
    }

    fn field_path_offset<I: Iterator<Item = usize>>(&self, mut ty: Ty<'tcx>, path: I) -> EvalResult<Size> {
        let mut offset = Size::from_bytes(0);

        // Skip the initial 0 intended for LLVM GEP.
        for field_index in path {
            let field_offset = self.get_field_offset(ty, field_index)?;
            ty = self.get_field_ty(ty, field_index)?;
            offset = offset.checked_add(field_offset, &self.tcx.data_layout).unwrap();
        }

        Ok(offset)
    }

    fn get_field_ty(&self, ty: Ty<'tcx>, field_index: usize) -> EvalResult<Ty<'tcx>> {
        match ty.sty {
            ty::TyStruct(adt_def, substs) => {
                Ok(adt_def.struct_variant().fields[field_index].ty(self.tcx, substs))
            }

            ty::TyRef(_, ty::TypeAndMut { ty, .. }) |
            ty::TyRawPtr(ty::TypeAndMut { ty, .. }) |
            ty::TyBox(ty) => {
                assert_eq!(field_index, 0);
                Ok(ty)
            }
            _ => Err(EvalError::Unimplemented(format!("can't handle type: {:?}", ty))),
        }
    }

    fn get_field_offset(&self, ty: Ty<'tcx>, field_index: usize) -> EvalResult<Size> {
        let layout = self.type_layout(ty);

        use rustc::ty::layout::Layout::*;
        match *layout {
            Univariant { .. } => {
                assert_eq!(field_index, 0);
                Ok(Size::from_bytes(0))
            }
            FatPointer { .. } => {
                let bytes = layout::FAT_PTR_ADDR * self.memory.pointer_size;
                Ok(Size::from_bytes(bytes as u64))
            }
            _ => Err(EvalError::Unimplemented(format!("can't handle type: {:?}, with layout: {:?}", ty, layout))),
        }
    }

    fn eval_operand(&mut self, op: &mir::Operand<'tcx>) -> EvalResult<Pointer> {
        use rustc::mir::repr::Operand::*;
        match *op {
            Consume(ref lvalue) => Ok(self.eval_lvalue(lvalue)?.to_ptr()),
            Constant(mir::Constant { ref literal, ty, .. }) => {
                use rustc::mir::repr::Literal::*;
                match *literal {
                    Value { ref value } => Ok(self.const_to_ptr(value)?),
                    Item { def_id, substs } => {
                        if let ty::TyFnDef(..) = ty.sty {
                            // function items are zero sized
                            Ok(self.memory.allocate(0))
                        } else {
                            let cid = ConstantId {
                                def_id: def_id,
                                substs: substs,
                                kind: ConstantKind::Global,
                            };
                            Ok(*self.statics.get(&cid).expect("static should have been cached (rvalue)"))
                        }
                    },
                    Promoted { index } => {
                        let cid = ConstantId {
                            def_id: self.frame().def_id,
                            substs: self.substs(),
                            kind: ConstantKind::Promoted(index),
                        };
                        Ok(*self.statics.get(&cid).expect("a promoted constant hasn't been precomputed"))
                    },
                }
            }
        }
    }

    fn eval_lvalue(&mut self, lvalue: &mir::Lvalue<'tcx>) -> EvalResult<Lvalue> {
        use rustc::mir::repr::Lvalue::*;
        let ptr = match *lvalue {
            ReturnPointer => self.frame().return_ptr
                .expect("ReturnPointer used in a function with no return value"),
            Arg(i) => self.frame().locals[i.index()],
            Var(i) => self.frame().locals[self.frame().var_offset + i.index()],
            Temp(i) => self.frame().locals[self.frame().temp_offset + i.index()],

            Static(def_id) => {
                let substs = self.tcx.mk_substs(subst::Substs::empty());
                let cid = ConstantId {
                    def_id: def_id,
                    substs: substs,
                    kind: ConstantKind::Global,
                };
                *self.statics.get(&cid).expect("static should have been cached (lvalue)")
            },

            Projection(ref proj) => {
                let base = self.eval_lvalue(&proj.base)?;
                let base_ty = self.lvalue_ty(&proj.base);
                let base_layout = self.type_layout(base_ty);

                use rustc::mir::repr::ProjectionElem::*;
                match proj.elem {
                    Field(field, _) => {
                        use rustc::ty::layout::Layout::*;
                        let variant = match *base_layout {
                            Univariant { ref variant, .. } => variant,
                            General { ref variants, .. } => {
                                if let LvalueExtra::DowncastVariant(variant_idx) = base.extra {
                                    &variants[variant_idx]
                                } else {
                                    panic!("field access on enum had no variant index");
                                }
                            }
                            RawNullablePointer { .. } => {
                                assert_eq!(field.index(), 0);
                                return Ok(base);
                            }
                            StructWrappedNullablePointer { ref nonnull, .. } => nonnull,
                            _ => panic!("field access on non-product type: {:?}", base_layout),
                        };

                        let offset = variant.field_offset(field.index()).bytes();
                        base.ptr.offset(offset as isize)
                    },

                    Downcast(_, variant) => {
                        use rustc::ty::layout::Layout::*;
                        match *base_layout {
                            General { discr, .. } => {
                                return Ok(Lvalue {
                                    ptr: base.ptr.offset(discr.size().bytes() as isize),
                                    extra: LvalueExtra::DowncastVariant(variant),
                                });
                            }
                            RawNullablePointer { .. } | StructWrappedNullablePointer { .. } => {
                                return Ok(base);
                            }
                            _ => panic!("variant downcast on non-aggregate: {:?}", base_layout),
                        }
                    },

                    Deref => {
                        let pointee_ty = pointee_type(base_ty).expect("Deref of non-pointer");
                        let ptr = self.memory.read_ptr(base.ptr)?;
                        let extra = match pointee_ty.sty {
                            ty::TySlice(_) | ty::TyStr => {
                                let len_ptr = base.ptr.offset(self.memory.pointer_size as isize);
                                let len = self.memory.read_usize(len_ptr)?;
                                LvalueExtra::Length(len)
                            }
                            ty::TyTrait(_) => unimplemented!(),
                            _ => LvalueExtra::None,
                        };
                        return Ok(Lvalue { ptr: ptr, extra: extra });
                    }

                    Index(ref operand) => {
                        let elem_size = match base_ty.sty {
                            ty::TyArray(elem_ty, _) |
                            ty::TySlice(elem_ty) => self.type_size(elem_ty),
                            _ => panic!("indexing expected an array or slice, got {:?}", base_ty),
                        };
                        let n_ptr = self.eval_operand(operand)?;
                        let n = self.memory.read_usize(n_ptr)?;
                        base.ptr.offset(n as isize * elem_size as isize)
                    }

                    ConstantIndex { .. } => unimplemented!(),
                    Subslice { .. } => unimplemented!(),
                }
            }
        };

        Ok(Lvalue { ptr: ptr, extra: LvalueExtra::None })
    }

    fn lvalue_ty(&self, lvalue: &mir::Lvalue<'tcx>) -> Ty<'tcx> {
        self.monomorphize(self.mir().lvalue_ty(self.tcx, lvalue).to_ty(self.tcx), self.substs())
    }

    fn operand_ty(&self, operand: &mir::Operand<'tcx>) -> Ty<'tcx> {
        self.monomorphize(self.mir().operand_ty(self.tcx, operand), self.substs())
    }

    fn move_(&mut self, src: Pointer, dest: Pointer, ty: Ty<'tcx>) -> EvalResult<()> {
        let size = self.type_size(ty);
        self.memory.copy(src, dest, size)?;
        if self.type_needs_drop(ty) {
            self.memory.drop_fill(src, size)?;
        }
        Ok(())
    }

    pub fn read_primval(&mut self, ptr: Pointer, ty: Ty<'tcx>) -> EvalResult<PrimVal> {
        use syntax::ast::{IntTy, UintTy};
        let val = match (self.memory.pointer_size, &ty.sty) {
            (_, &ty::TyBool)              => PrimVal::Bool(self.memory.read_bool(ptr)?),
            (_, &ty::TyInt(IntTy::I8))    => PrimVal::I8(self.memory.read_int(ptr, 1)? as i8),
            (2, &ty::TyInt(IntTy::Is)) |
            (_, &ty::TyInt(IntTy::I16))   => PrimVal::I16(self.memory.read_int(ptr, 2)? as i16),
            (4, &ty::TyInt(IntTy::Is)) |
            (_, &ty::TyInt(IntTy::I32))   => PrimVal::I32(self.memory.read_int(ptr, 4)? as i32),
            (8, &ty::TyInt(IntTy::Is)) |
            (_, &ty::TyInt(IntTy::I64))   => PrimVal::I64(self.memory.read_int(ptr, 8)? as i64),
            (_, &ty::TyUint(UintTy::U8))  => PrimVal::U8(self.memory.read_uint(ptr, 1)? as u8),
            (2, &ty::TyUint(UintTy::Us)) |
            (_, &ty::TyUint(UintTy::U16)) => PrimVal::U16(self.memory.read_uint(ptr, 2)? as u16),
            (4, &ty::TyUint(UintTy::Us)) |
            (_, &ty::TyUint(UintTy::U32)) => PrimVal::U32(self.memory.read_uint(ptr, 4)? as u32),
            (8, &ty::TyUint(UintTy::Us)) |
            (_, &ty::TyUint(UintTy::U64)) => PrimVal::U64(self.memory.read_uint(ptr, 8)? as u64),

            (_, &ty::TyRef(_, ty::TypeAndMut { ty, .. })) |
            (_, &ty::TyRawPtr(ty::TypeAndMut { ty, .. })) => {
                if self.type_is_sized(ty) {
                    match self.memory.read_ptr(ptr) {
                        Ok(p) => PrimVal::AbstractPtr(p),
                        Err(EvalError::ReadBytesAsPointer) => {
                            PrimVal::IntegerPtr(self.memory.read_usize(ptr)?)
                        }
                        Err(e) => return Err(e),
                    }
                } else {
                    return Err(EvalError::Unimplemented(format!("unimplemented: primitive read of fat pointer type: {:?}", ty)));
                }
            }

            _ => panic!("primitive read of non-primitive type: {:?}", ty),
        };
        Ok(val)
    }

    fn frame(&self) -> &Frame<'a, 'tcx> {
        self.stack.last().expect("no call frames exist")
    }

    fn frame_mut(&mut self) -> &mut Frame<'a, 'tcx> {
        self.stack.last_mut().expect("no call frames exist")
    }

    fn mir(&self) -> CachedMir<'a, 'tcx> {
        self.frame().mir.clone()
    }

    fn substs(&self) -> &'tcx Substs<'tcx> {
        self.frame().substs
    }
}

fn pointee_type(ptr_ty: ty::Ty) -> Option<ty::Ty> {
    match ptr_ty.sty {
        ty::TyRef(_, ty::TypeAndMut { ty, .. }) |
        ty::TyRawPtr(ty::TypeAndMut { ty, .. }) |
        ty::TyBox(ty) => {
            Some(ty)
        }
        _ => None,
    }
}

impl Lvalue {
    fn to_ptr(self) -> Pointer {
        assert_eq!(self.extra, LvalueExtra::None);
        self.ptr
    }
}

impl<'mir, 'tcx: 'mir> Deref for CachedMir<'mir, 'tcx> {
    type Target = mir::Mir<'tcx>;
    fn deref(&self) -> &mir::Mir<'tcx> {
        match *self {
            CachedMir::Ref(r) => r,
            CachedMir::Owned(ref rc) => rc,
        }
    }
}

#[derive(Debug)]
pub struct ImplMethod<'tcx> {
    pub method: Rc<ty::Method<'tcx>>,
    pub substs: &'tcx Substs<'tcx>,
    pub is_provided: bool,
}

/// Locates the applicable definition of a method, given its name.
pub fn get_impl_method<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    impl_def_id: DefId,
    substs: &'tcx Substs<'tcx>,
    name: ast::Name,
) -> ImplMethod<'tcx> {
    assert!(!substs.types.needs_infer());

    let trait_def_id = tcx.trait_id_of_impl(impl_def_id).unwrap();
    let trait_def = tcx.lookup_trait_def(trait_def_id);

    match trait_def.ancestors(impl_def_id).fn_defs(tcx, name).next() {
        Some(node_item) => {
            let substs = tcx.normalizing_infer_ctxt(ProjectionMode::Any).enter(|infcx| {
                let substs = traits::translate_substs(&infcx, impl_def_id,
                                                      substs, node_item.node);
                tcx.lift(&substs).unwrap_or_else(|| {
                    bug!("trans::meth::get_impl_method: translate_substs \
                          returned {:?} which contains inference types/regions",
                         substs);
                })
            });
            ImplMethod {
                method: node_item.item,
                substs: substs,
                is_provided: node_item.node.is_from_trait(),
            }
        }
        None => {
            bug!("method {:?} not found in {:?}", name, impl_def_id)
        }
    }
}

// TODO(solson): Upstream these methods into rustc::ty::layout.

trait IntegerExt {
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
        }
    }
}

trait StructExt {
    fn field_offset(&self, index: usize) -> Size;
}

impl StructExt for layout::Struct {
    fn field_offset(&self, index: usize) -> Size {
        if index == 0 {
            Size::from_bytes(0)
        } else {
            self.offset_after_field[index - 1]
        }
    }
}
