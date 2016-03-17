use rustc::middle::const_eval;
use rustc::middle::def_id::DefId;
use rustc::middle::subst::{self, Subst, Substs};
use rustc::middle::traits;
use rustc::middle::ty::{self, TyCtxt};
use rustc::mir::mir_map::MirMap;
use rustc::mir::repr as mir;
use rustc::util::nodemap::DefIdMap;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use syntax::codemap::DUMMY_SP;

use error::EvalResult;
use memory::{self, FieldRepr, Memory, Pointer, Repr};
use primval::{self, PrimVal};

const TRACE_EXECUTION: bool = true;

struct Interpreter<'a, 'tcx: 'a> {
    /// The results of the type checker, from rustc.
    tcx: &'a TyCtxt<'tcx>,

    /// A mapping from NodeIds to Mir, from rustc. Only contains MIR for crate-local items.
    mir_map: &'a MirMap<'tcx>,

    /// A local cache from DefIds to Mir for non-crate-local items.
    mir_cache: RefCell<DefIdMap<Rc<mir::Mir<'tcx>>>>,

    /// The virtual memory system.
    memory: Memory,

    /// The virtual call stack.
    stack: Vec<Frame<'a, 'tcx>>,

    /// Another stack containing the type substitutions for the current function invocation. It
    /// exists separately from `stack` because it must contain the `Substs` for a function while
    /// *creating* the `Frame` for that same function.
    substs_stack: Vec<&'tcx Substs<'tcx>>,
}

/// A stack frame.
struct Frame<'a, 'tcx: 'a> {
    /// The MIR for the function called on this frame.
    mir: CachedMir<'a, 'tcx>,

    /// The block this frame will execute when a function call returns back to this frame.
    next_block: mir::BasicBlock,

    /// A pointer for writing the return value of the current call if it's not a diverging call.
    return_ptr: Option<Pointer>,

    /// The list of locals for the current function, stored in order as
    /// `[arguments..., variables..., temporaries...]`. The variables begin at `self.var_offset`
    /// and the temporaries at `self.temp_offset`.
    locals: Vec<Pointer>,

    /// The offset of the first variable in `self.locals`.
    var_offset: usize,

    /// The offset of the first temporary in `self.locals`.
    temp_offset: usize,
}

#[derive(Clone)]
enum CachedMir<'mir, 'tcx: 'mir> {
    Ref(&'mir mir::Mir<'tcx>),
    Owned(Rc<mir::Mir<'tcx>>)
}

/// Represents the action to be taken in the main loop as a result of executing a terminator.
enum TerminatorTarget {
    /// Make a local jump to the given block.
    Block(mir::BasicBlock),

    /// Start executing from the new current frame. (For function calls.)
    Call,

    /// Stop executing the current frame and resume the previous frame.
    Return,
}

impl<'a, 'tcx: 'a> Interpreter<'a, 'tcx> {
    fn new(tcx: &'a TyCtxt<'tcx>, mir_map: &'a MirMap<'tcx>) -> Self {
        Interpreter {
            tcx: tcx,
            mir_map: mir_map,
            mir_cache: RefCell::new(DefIdMap()),
            memory: Memory::new(),
            stack: Vec::new(),
            substs_stack: Vec::new(),
        }
    }

    fn run(&mut self) -> EvalResult<()> {
        use std::fmt::Debug;
        fn print_trace<T: Debug>(t: &T, suffix: &'static str, indent: usize) {
            if !TRACE_EXECUTION { return; }
            for _ in 0..indent { print!("  "); }
            println!("{:?}{}", t, suffix);
        }

        'outer: while !self.stack.is_empty() {
            let mut current_block = self.current_frame().next_block;

            loop {
                print_trace(&current_block, ":", self.stack.len());
                let current_mir = self.current_frame().mir.clone(); // Cloning a reference.
                let block_data = current_mir.basic_block_data(current_block);

                for stmt in &block_data.statements {
                    print_trace(stmt, "", self.stack.len() + 1);
                    let mir::StatementKind::Assign(ref lvalue, ref rvalue) = stmt.kind;
                    try!(self.eval_assignment(lvalue, rvalue));
                }

                let terminator = block_data.terminator();
                print_trace(terminator, "", self.stack.len() + 1);

                match try!(self.eval_terminator(terminator)) {
                    TerminatorTarget::Block(block) => current_block = block,
                    TerminatorTarget::Return => {
                        self.pop_stack_frame();
                        self.substs_stack.pop();
                        continue 'outer;
                    }
                    TerminatorTarget::Call => continue 'outer,
                }
            }
        }

        Ok(())
    }

    fn push_stack_frame(&mut self, mir: CachedMir<'a, 'tcx>, args: &[mir::Operand<'tcx>],
                        return_ptr: Option<Pointer>) -> EvalResult<()> {
        let num_args = mir.arg_decls.len();
        let num_vars = mir.var_decls.len();
        let num_temps = mir.temp_decls.len();
        assert_eq!(args.len(), num_args);

        let mut locals = Vec::with_capacity(num_args + num_vars + num_temps);

        for (arg_decl, arg_operand) in mir.arg_decls.iter().zip(args) {
            let repr = self.ty_to_repr(arg_decl.ty);
            let dest = self.memory.allocate(repr.size());
            let src = try!(self.eval_operand(arg_operand));
            try!(self.memory.copy(src, dest, repr.size()));
            locals.push(dest);
        }

        let var_tys = mir.var_decls.iter().map(|v| v.ty);
        let temp_tys = mir.temp_decls.iter().map(|t| t.ty);
        locals.extend(var_tys.chain(temp_tys).map(|ty| {
            let repr = self.ty_to_repr(ty).size();
            self.memory.allocate(repr)
        }));

        self.stack.push(Frame {
            mir: mir.clone(),
            next_block: mir::START_BLOCK,
            return_ptr: return_ptr,
            locals: locals,
            var_offset: num_args,
            temp_offset: num_args + num_vars,
        });

        Ok(())
    }

    fn pop_stack_frame(&mut self) {
        let _frame = self.stack.pop().expect("tried to pop a stack frame, but there were none");
        // TODO(tsion): Deallocate local variables.
    }

    fn eval_terminator(&mut self, terminator: &mir::Terminator<'tcx>)
            -> EvalResult<TerminatorTarget> {
        use rustc::mir::repr::Terminator::*;
        let target = match *terminator {
            Return => TerminatorTarget::Return,

            Goto { target } => TerminatorTarget::Block(target),

            If { ref cond, targets: (then_target, else_target) } => {
                let cond_ptr = try!(self.eval_operand(cond));
                let cond_val = try!(self.memory.read_bool(cond_ptr));
                TerminatorTarget::Block(if cond_val { then_target } else { else_target })
            }

            SwitchInt { ref discr, ref values, ref targets, .. } => {
                let discr_ptr = try!(self.eval_lvalue(discr));
                let discr_repr = self.lvalue_repr(discr);
                let discr_val = try!(self.memory.read_primval(discr_ptr, &discr_repr));

                // Branch to the `otherwise` case by default, if no match is found.
                let mut target_block = targets[targets.len() - 1];

                for (index, val_const) in values.iter().enumerate() {
                    let ptr = try!(self.const_to_ptr(val_const));
                    let val = try!(self.memory.read_primval(ptr, &discr_repr));
                    if discr_val == val {
                        target_block = targets[index];
                        break;
                    }
                }

                TerminatorTarget::Block(target_block)
            }

            Switch { ref discr, ref targets, .. } => {
                let adt_ptr = try!(self.eval_lvalue(discr));
                let adt_repr = self.lvalue_repr(discr);
                let discr_repr = match adt_repr {
                    Repr::Sum { ref discr, .. } => discr,
                    _ => panic!("attmpted to switch on non-sum type"),
                };
                let discr_val = try!(self.memory.read_primval(adt_ptr, &discr_repr));
                TerminatorTarget::Block(targets[discr_val.to_usize()])
            }

            Call { ref func, ref args, ref destination, .. } => {
                let mut return_ptr = None;
                if let Some((ref lv, target)) = *destination {
                    self.current_frame_mut().next_block = target;
                    return_ptr = Some(try!(self.eval_lvalue(lv)));
                }

                let func_ty = self.current_frame().mir.operand_ty(self.tcx, func);

                match func_ty.sty {
                    ty::TyFnDef(def_id, substs, fn_ty) => {
                        let substs = self.tcx.mk_substs(
                            substs.subst(self.tcx, self.current_substs()));

                        use syntax::abi::Abi;
                        match fn_ty.abi {
                            Abi::RustIntrinsic => {
                                let ret_ptr = &mir::Lvalue::ReturnPointer;
                                let dest = try!(self.eval_lvalue(ret_ptr));
                                let dest_repr = self.lvalue_repr(ret_ptr);

                                match &self.tcx.item_name(def_id).as_str()[..] {
                                    "size_of" => {
                                        let ty = *substs.types.get(subst::FnSpace, 0);
                                        let size = PrimVal::from_usize(
                                            self.ty_to_repr(ty).size(),
                                            &dest_repr
                                        );
                                        try!(self.memory.write_primval(dest, size));
                                    }

                                    "offset" => {
                                        let pointee_ty = *substs.types.get(subst::FnSpace, 0);
                                        let pointee_size = self.ty_to_repr(pointee_ty).size() as isize;
                                        let ptr_arg = try!(self.eval_operand(&args[0]));
                                        let offset_arg = try!(self.eval_operand(&args[1]));
                                        let ptr = try!(self.memory.read_ptr(ptr_arg));
                                        // TODO(tsion): read_isize
                                        let offset = try!(self.memory.read_i64(offset_arg));
                                        let result_ptr = ptr.offset(offset as isize * pointee_size);
                                        try!(self.memory.write_ptr(dest, result_ptr));
                                    }

                                    name => panic!("can't handle intrinsic named {}", name),
                                }

                                // Since we pushed no stack frame, the main loop will act
                                // as if the call just completed and it's returning to the
                                // current frame.
                                TerminatorTarget::Call
                            }

                            Abi::Rust => {
                                // Only trait methods can have a Self parameter.
                                let (def_id, substs) = if substs.self_ty().is_some() {
                                    self.trait_method(def_id, substs)
                                } else {
                                    (def_id, substs)
                                };

                                let mir = self.load_mir(def_id);
                                self.substs_stack.push(substs);
                                try!(self.push_stack_frame(mir, args, return_ptr));
                                TerminatorTarget::Call
                            }

                            abi => panic!("can't handle function with ABI {:?}", abi),
                        }
                    }

                    _ => panic!("can't handle callee of type {:?}", func_ty),
                }
            }

            Drop { target, .. } => {
                // TODO: Handle destructors and dynamic drop.
                TerminatorTarget::Block(target)
            }

            Resume => unimplemented!(),
        };

        Ok(target)
    }

    fn assign_to_product(&mut self, dest: Pointer, dest_repr: &Repr,
                         operands: &[mir::Operand<'tcx>]) -> EvalResult<()> {
        match *dest_repr {
            Repr::Product { ref fields, .. } => {
                for (field, operand) in fields.iter().zip(operands) {
                    let src = try!(self.eval_operand(operand));
                    let field_dest = dest.offset(field.offset as isize);
                    try!(self.memory.copy(src, field_dest, field.repr.size()));
                }
            }
            _ => panic!("expected Repr::Product target"),
        }
        Ok(())
    }

    fn eval_assignment(&mut self, lvalue: &mir::Lvalue<'tcx>, rvalue: &mir::Rvalue<'tcx>)
        -> EvalResult<()>
    {
        let dest = try!(self.eval_lvalue(lvalue));
        let dest_repr = self.lvalue_repr(lvalue);

        use rustc::mir::repr::Rvalue::*;
        match *rvalue {
            Use(ref operand) => {
                let src = try!(self.eval_operand(operand));
                self.memory.copy(src, dest, dest_repr.size())
            }

            BinaryOp(bin_op, ref left, ref right) => {
                let (left_ptr, left_repr) = try!(self.eval_operand_and_repr(left));
                let (right_ptr, right_repr) = try!(self.eval_operand_and_repr(right));
                let left_val = try!(self.memory.read_primval(left_ptr, &left_repr));
                let right_val = try!(self.memory.read_primval(right_ptr, &right_repr));
                self.memory.write_primval(dest, primval::binary_op(bin_op, left_val, right_val))
            }

            UnaryOp(un_op, ref operand) => {
                let (ptr, repr) = try!(self.eval_operand_and_repr(operand));
                let val = try!(self.memory.read_primval(ptr, &repr));
                self.memory.write_primval(dest, primval::unary_op(un_op, val))
            }

            Aggregate(ref kind, ref operands) => {
                use rustc::mir::repr::AggregateKind::*;
                match *kind {
                    Tuple => self.assign_to_product(dest, &dest_repr, operands),

                    Adt(ref adt_def, variant_idx, _) => match adt_def.adt_kind() {
                        ty::AdtKind::Struct => self.assign_to_product(dest, &dest_repr, operands),

                        ty::AdtKind::Enum => match dest_repr {
                            Repr::Sum { ref discr, ref variants, .. } => {
                                if discr.size() > 0 {
                                    let discr_val = PrimVal::from_usize(variant_idx, discr);
                                    try!(self.memory.write_primval(dest, discr_val));
                                }
                                self.assign_to_product(
                                    dest.offset(discr.size() as isize),
                                    &variants[variant_idx],
                                    operands
                                )
                            }
                            _ => panic!("expected Repr::Sum target"),
                        }
                    },

                    Vec => match dest_repr {
                        Repr::Array { ref elem, length } => {
                            assert_eq!(length, operands.len());
                            let elem_size = elem.size();
                            for (i, operand) in operands.iter().enumerate() {
                                let src = try!(self.eval_operand(operand));
                                let offset = i * elem_size;
                                let elem_dest = dest.offset(offset as isize);
                                try!(self.memory.copy(src, elem_dest, elem_size));
                            }
                            Ok(())
                        }
                        _ => panic!("expected Repr::Array target"),
                    },

                    Closure(..) => unimplemented!(),
                }
            }

            Ref(_, _, ref lvalue) => {
                let ptr = try!(self.eval_lvalue(lvalue));
                self.memory.write_ptr(dest, ptr)
            }

            Box(ty) => {
                let repr = self.ty_to_repr(ty);
                let ptr = self.memory.allocate(repr.size());
                self.memory.write_ptr(dest, ptr)
            }

            Cast(kind, ref operand, dest_ty) => {
                fn pointee_type<'tcx>(ptr_ty: ty::Ty<'tcx>) -> Option<ty::Ty<'tcx>> {
                    match ptr_ty.sty {
                        ty::TyRef(_, ty::TypeAndMut { ty, .. }) |
                        ty::TyRawPtr(ty::TypeAndMut { ty, .. }) |
                        ty::TyBox(ty) => {
                            Some(ty)
                        }

                        _ => None,
                    }
                }

                let src = try!(self.eval_operand(operand));
                let src_ty = self.current_frame().mir.operand_ty(self.tcx, operand);

                use rustc::mir::repr::CastKind::*;
                match kind {
                    Unsize => {
                        try!(self.memory.copy(src, dest, 8));
                        let src_pointee_ty = pointee_type(src_ty).unwrap();
                        let dest_pointee_ty = pointee_type(dest_ty).unwrap();

                        match (&src_pointee_ty.sty, &dest_pointee_ty.sty) {
                            (&ty::TyArray(_, length), &ty::TySlice(_)) =>
                                // TODO(tsion): Add write_usize? (Host/target issues.)
                                self.memory.write_u64(dest.offset(8), length as u64),

                            _ => panic!("can't handle cast: {:?}", rvalue),
                        }
                    }

                    Misc => {
                        if pointee_type(src_ty).is_some() && pointee_type(dest_ty).is_some() {
                            self.memory.copy(src, dest, 8)
                        } else {
                            self.memory.copy(src, dest, 8)
                            // panic!("can't handle cast: {:?}", rvalue);
                        }
                    }

                    _ => panic!("can't handle cast: {:?}", rvalue),
                }
            }

            ref r => panic!("can't handle rvalue: {:?}", r),
        }
    }

    fn eval_operand(&mut self, op: &mir::Operand<'tcx>) -> EvalResult<Pointer> {
        self.eval_operand_and_repr(op).map(|(p, _)| p)
    }

    fn eval_operand_and_repr(&mut self, op: &mir::Operand<'tcx>) -> EvalResult<(Pointer, Repr)> {
        use rustc::mir::repr::Operand::*;
        match *op {
            Consume(ref lvalue) => Ok((try!(self.eval_lvalue(lvalue)), self.lvalue_repr(lvalue))),
            Constant(mir::Constant { ref literal, ty, .. }) => {
                use rustc::mir::repr::Literal::*;
                match *literal {
                    Value { ref value } => Ok((
                        try!(self.const_to_ptr(value)),
                        self.ty_to_repr(ty),
                    )),
                    ref l => panic!("can't handle item literal: {:?}", l),
                }
            }
        }
    }

    fn lvalue_repr(&self, lvalue: &mir::Lvalue<'tcx>) -> Repr {
        use rustc::mir::tcx::LvalueTy;
        match self.current_frame().mir.lvalue_ty(self.tcx, lvalue) {
            LvalueTy::Ty { ty } => self.ty_to_repr(ty),
            LvalueTy::Downcast { ref adt_def, substs, variant_index } =>
                self.make_variant_repr(&adt_def.variants[variant_index], substs),
        }
    }

    fn eval_lvalue(&self, lvalue: &mir::Lvalue<'tcx>) -> EvalResult<Pointer> {
        let frame = self.current_frame();

        use rustc::mir::repr::Lvalue::*;
        let ptr = match *lvalue {
            ReturnPointer =>
                frame.return_ptr.expect("ReturnPointer used in a function with no return value"),
            Arg(i) => frame.locals[i as usize],
            Var(i) => frame.locals[frame.var_offset + i as usize],
            Temp(i) => frame.locals[frame.temp_offset + i as usize],

            Projection(ref proj) => {
                let base_ptr = try!(self.eval_lvalue(&proj.base));
                let base_repr = self.lvalue_repr(&proj.base);
                use rustc::mir::repr::ProjectionElem::*;
                match proj.elem {
                    Field(field, _) => match base_repr {
                        Repr::Product { ref fields, .. } =>
                            base_ptr.offset(fields[field.index()].offset as isize),
                        _ => panic!("field access on non-product type: {:?}", base_repr),
                    },

                    Downcast(..) => match base_repr {
                        Repr::Sum { ref discr, .. } => base_ptr.offset(discr.size() as isize),
                        _ => panic!("variant downcast on non-sum type"),
                    },

                    Deref => try!(self.memory.read_ptr(base_ptr)),

                    _ => unimplemented!(),
                }
            }

            ref l => panic!("can't handle lvalue: {:?}", l),
        };

        Ok(ptr)
    }

    fn const_to_ptr(&mut self, const_val: &const_eval::ConstVal) -> EvalResult<Pointer> {
        use rustc::middle::const_eval::ConstVal::*;
        match *const_val {
            Float(_f) => unimplemented!(),
            Integral(int) => {
                // TODO(tsion): Check int constant type.
                let ptr = self.memory.allocate(8);
                try!(self.memory.write_u64(ptr, int.to_u64_unchecked()));
                Ok(ptr)
            }
            Str(ref _s) => unimplemented!(),
            ByteStr(ref _bs) => unimplemented!(),
            Bool(b) => {
                let ptr = self.memory.allocate(Repr::Bool.size());
                try!(self.memory.write_bool(ptr, b));
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

    fn make_product_repr<I>(&self, iter: I) -> Repr where I: IntoIterator<Item = ty::Ty<'tcx>> {
        let mut size = 0;
        let fields = iter.into_iter().map(|ty| {
            let repr = self.ty_to_repr(ty);
            let old_size = size;
            size += repr.size();
            FieldRepr { offset: old_size, repr: repr }
        }).collect();
        Repr::Product { size: size, fields: fields }
    }

    fn make_variant_repr(&self, v: ty::VariantDef<'tcx>, substs: &'tcx Substs<'tcx>) -> Repr {
        let field_tys = v.fields.iter().map(|f| f.ty(self.tcx, substs));
        self.make_product_repr(field_tys)
    }

    // TODO(tsion): Cache these outputs.
    fn ty_to_repr(&self, ty: ty::Ty<'tcx>) -> Repr {
        use syntax::ast::{IntTy, UintTy};
        match ty.subst(self.tcx, self.current_substs()).sty {
            ty::TyBool => Repr::Bool,

            ty::TyInt(IntTy::Is)  => Repr::isize(),
            ty::TyInt(IntTy::I8)  => Repr::I8,
            ty::TyInt(IntTy::I16) => Repr::I16,
            ty::TyInt(IntTy::I32) => Repr::I32,
            ty::TyInt(IntTy::I64) => Repr::I64,

            ty::TyUint(UintTy::Us)  => Repr::usize(),
            ty::TyUint(UintTy::U8)  => Repr::U8,
            ty::TyUint(UintTy::U16) => Repr::U16,
            ty::TyUint(UintTy::U32) => Repr::U32,
            ty::TyUint(UintTy::U64) => Repr::U64,

            ty::TyTuple(ref fields) => self.make_product_repr(fields.iter().cloned()),

            ty::TyEnum(adt_def, substs) => {
                let num_variants = adt_def.variants.len();

                let discr = if num_variants <= 1 {
                    Repr::Product { size: 0, fields: vec![] }
                } else if num_variants <= 1 << 8 {
                    Repr::U8
                } else if num_variants <= 1 << 16 {
                    Repr::U16
                } else if num_variants <= 1 << 32 {
                    Repr::U32
                } else {
                    Repr::U64
                };

                let variants: Vec<Repr> = adt_def.variants.iter().map(|v| {
                    self.make_variant_repr(v, substs)
                }).collect();

                Repr::Sum {
                    discr: Box::new(discr),
                    max_variant_size: variants.iter().map(Repr::size).max().unwrap_or(0),
                    variants: variants,
                }
            }

            ty::TyStruct(adt_def, substs) => {
                assert_eq!(adt_def.variants.len(), 1);
                self.make_variant_repr(&adt_def.variants[0], substs)
            }

            ty::TyArray(ref elem_ty, length) => Repr::Array {
                elem: Box::new(self.ty_to_repr(elem_ty)),
                length: length,
            },

            ty::TyRef(_, ty::TypeAndMut { ty, .. }) |
            ty::TyRawPtr(ty::TypeAndMut { ty, .. }) |
            ty::TyBox(ty) => {
                if ty.is_sized(&self.tcx.empty_parameter_environment(), DUMMY_SP) {
                    Repr::Pointer
                } else {
                    Repr::FatPointer
                }
            }

            ref t => panic!("can't convert type to repr: {:?}", t),
        }
    }

    fn current_frame(&self) -> &Frame<'a, 'tcx> {
        self.stack.last().expect("no call frames exist")
    }

    fn current_frame_mut(&mut self) -> &mut Frame<'a, 'tcx> {
        self.stack.last_mut().expect("no call frames exist")
    }

    fn current_substs(&self) -> &'tcx Substs<'tcx> {
        self.substs_stack.last().cloned().unwrap_or_else(|| self.tcx.mk_substs(Substs::empty()))
    }

    fn load_mir(&self, def_id: DefId) -> CachedMir<'a, 'tcx> {
        match self.tcx.map.as_local_node_id(def_id) {
            Some(node_id) => CachedMir::Ref(self.mir_map.map.get(&node_id).unwrap()),
            None => {
                let mut mir_cache = self.mir_cache.borrow_mut();
                if let Some(mir) = mir_cache.get(&def_id) {
                    return CachedMir::Owned(mir.clone());
                }

                use rustc::middle::cstore::CrateStore;
                let cs = &self.tcx.sess.cstore;
                let mir = cs.maybe_get_item_mir(self.tcx, def_id).unwrap();
                let cached = Rc::new(mir);
                mir_cache.insert(def_id, cached.clone());
                CachedMir::Owned(cached)
            }
        }
    }

    fn fulfill_obligation(&self, trait_ref: ty::PolyTraitRef<'tcx>) -> traits::Vtable<'tcx, ()> {
        use rustc::middle::infer;
        use syntax::ast;

        // Do the initial selection for the obligation. This yields the shallow result we are
        // looking for -- that is, what specific impl.
        let infcx = infer::normalizing_infer_ctxt(self.tcx, &self.tcx.tables);
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
        let vtable = infer::drain_fulfillment_cx_or_panic(
            DUMMY_SP, &infcx, &mut fulfill_cx, &vtable
        );

        vtable
    }

    /// Trait method, which has to be resolved to an impl method.
    pub fn trait_method(&self, def_id: DefId, substs: &'tcx Substs<'tcx>)
            -> (DefId, &'tcx Substs<'tcx>) {
        let method_item = self.tcx.impl_or_trait_item(def_id);
        let trait_id = method_item.container().id();
        let trait_ref = ty::Binder(substs.to_trait_ref(self.tcx, trait_id));
        match self.fulfill_obligation(trait_ref) {
            traits::VtableImpl(vtable_impl) => {
                let impl_did = vtable_impl.impl_def_id;
                let mname = self.tcx.item_name(def_id);
                // Create a concatenated set of substitutions which includes those from the
                // impl and those from the method:
                let impl_substs = vtable_impl.substs.with_method_from(&substs);
                let substs = self.tcx.mk_substs(impl_substs);
                let mth = self.tcx.get_impl_method(impl_did, substs, mname);

                println!("{:?} {:?}", mth.method.def_id, mth.substs);
                (mth.method.def_id, mth.substs)
            }
            traits::VtableClosure(vtable_closure) => {
                // The substitutions should have no type parameters remaining after passing
                // through fulfill_obligation
                let trait_closure_kind = self.tcx.lang_items.fn_trait_kind(trait_id).unwrap();
                unimplemented!()
                // vtable_closure.closure_def_id
                // vtable_closure.substs
                // trait_closure_kind

                // let method_ty = def_ty(tcx, def_id, substs);
                // let fn_ptr_ty = match method_ty.sty {
                //     ty::TyFnDef(_, _, fty) => tcx.mk_ty(ty::TyFnPtr(fty)),
                //     _ => unreachable!("expected fn item type, found {}",
                //                       method_ty)
                // };
                // Callee::ptr(immediate_rvalue(llfn, fn_ptr_ty))
            }
            traits::VtableFnPointer(fn_ty) => {
                let trait_closure_kind = self.tcx.lang_items.fn_trait_kind(trait_id).unwrap();
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
            traits::VtableObject(ref data) => {
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
}

impl<'mir, 'tcx: 'mir> Deref for CachedMir<'mir, 'tcx> {
    type Target = mir::Mir<'tcx>;
    fn deref(&self) -> &mir::Mir<'tcx> {
        match *self {
            CachedMir::Ref(r) => r,
            CachedMir::Owned(ref rc) => &rc,
        }
    }
}

pub fn interpret_start_points<'tcx>(tcx: &TyCtxt<'tcx>, mir_map: &MirMap<'tcx>) {
    /// Print the given allocation and all allocations it depends on.
    fn print_allocation_tree(memory: &Memory, alloc_id: memory::AllocId) {
        let alloc = memory.get(alloc_id).unwrap();
        println!("  {:?}: {:?}", alloc_id, alloc);
        for &target_alloc in alloc.relocations.values() {
            print_allocation_tree(memory, target_alloc);
        }
    }

    for (&id, mir) in &mir_map.map {
        for attr in tcx.map.attrs(id) {
            use syntax::attr::AttrMetaMethods;
            if attr.check_name("miri_run") {
                let item = tcx.map.expect_item(id);

                println!("Interpreting: {}", item.name);

                let mut miri = Interpreter::new(tcx, mir_map);
                let return_ptr = match mir.return_ty {
                    ty::FnConverging(ty) => {
                        let repr = miri.ty_to_repr(ty).size();
                        Some(miri.memory.allocate(repr))
                    }
                    ty::FnDiverging => None,
                };
                miri.push_stack_frame(CachedMir::Ref(mir), &[], return_ptr).unwrap();
                miri.run().unwrap();

                if let Some(ret) = return_ptr {
                    println!("Result:");
                    print_allocation_tree(&miri.memory, ret.alloc_id);
                    println!("");
                }
            }
        }
    }
}
