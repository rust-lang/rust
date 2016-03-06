// TODO(tsion): Remove this.
#![allow(unused_imports, dead_code, unused_variables)]

use byteorder::{self, ByteOrder};
use rustc::middle::const_eval;
use rustc::middle::cstore::CrateStore;
use rustc::middle::def_id;
use rustc::middle::ty::{self, TyCtxt};
use rustc::mir::mir_map::MirMap;
use rustc::mir::repr::{self as mir, Mir};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::iter;
use syntax::ast::Attribute;
use syntax::attr::AttrMetaMethods;

use memory::{self, Pointer, Repr, Allocation};

const TRACE_EXECUTION: bool = true;

#[derive(Clone, Debug)]
pub enum EvalError {
    DanglingPointerDeref,
    PointerOutOfBounds,
}

pub type EvalResult<T> = Result<T, EvalError>;

impl Error for EvalError {
    fn description(&self) -> &str {
        match *self {
            EvalError::DanglingPointerDeref => "dangling pointer was dereferenced",
            EvalError::PointerOutOfBounds => "pointer offset outside bounds of allocation",
        }
    }

    fn cause(&self) -> Option<&Error> { None }
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

// #[derive(Clone, Debug, PartialEq)]
// enum Value {
//     Uninit,
//     Bool(bool),
//     Int(i64), // FIXME(tsion): Should be bit-width aware.
//     Pointer(Pointer),
//     Adt { variant: usize, data_ptr: Pointer },
//     Func(def_id::DefId),
// }

/// A stack frame.
#[derive(Debug)]
struct Frame {
    /// A pointer for writing the return value of the current call, if it's not a diverging call.
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

impl Frame {
    fn arg_ptr(&self, i: u32) -> Pointer {
        self.locals[i as usize].clone()
    }

    fn var_ptr(&self, i: u32) -> Pointer {
        self.locals[self.var_offset + i as usize].clone()
    }

    fn temp_ptr(&self, i: u32) -> Pointer {
        self.locals[self.temp_offset + i as usize].clone()
    }
}

struct Interpreter<'a, 'tcx: 'a> {
    tcx: &'a TyCtxt<'tcx>,
    mir_map: &'a MirMap<'tcx>,
    memory: memory::Memory,
    stack: Vec<Frame>,
}

impl<'a, 'tcx> Interpreter<'a, 'tcx> {
    fn new(tcx: &'a TyCtxt<'tcx>, mir_map: &'a MirMap<'tcx>) -> Self {
        Interpreter {
            tcx: tcx,
            mir_map: mir_map,
            memory: memory::Memory::new(),
            stack: Vec::new(),
        }
    }

    fn push_stack_frame(&mut self, mir: &Mir, args: &[&mir::Operand], return_ptr: Option<Pointer>)
        -> EvalResult<()>
    {
        let num_args = mir.arg_decls.len();
        let num_vars = mir.var_decls.len();
        assert_eq!(args.len(), num_args);

        let arg_tys = mir.arg_decls.iter().map(|a| a.ty);
        let var_tys = mir.var_decls.iter().map(|v| v.ty);
        let temp_tys = mir.temp_decls.iter().map(|t| t.ty);

        let locals: Vec<Pointer> = arg_tys.chain(var_tys).chain(temp_tys).map(|ty| {
            self.memory.allocate(Repr::from_ty(ty))
        }).collect();

        for (dest, operand) in locals[..num_args].iter().zip(args) {
            let src = try!(self.operand_to_ptr(operand));
            try!(self.memory.copy(&src, dest, dest.repr.size()));
        }

        self.stack.push(Frame { 
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

    fn call(&mut self, mir: &Mir, args: &[&mir::Operand], return_ptr: Option<Pointer>)
        -> EvalResult<()>
    {
        try!(self.push_stack_frame(mir, args, return_ptr));
        let mut block = mir::START_BLOCK;

        loop {
            if TRACE_EXECUTION { println!("Entering block: {:?}", block); }
            let block_data = mir.basic_block_data(block);

            for stmt in &block_data.statements {
                if TRACE_EXECUTION { println!("{:?}", stmt); }

                match stmt.kind {
                    mir::StatementKind::Assign(ref lvalue, ref rvalue) => {
                        let ptr = try!(self.lvalue_to_ptr(lvalue));
                        try!(self.eval_rvalue_into(rvalue, &ptr));
                    }
                }
            }

            if TRACE_EXECUTION { println!("{:?}", block_data.terminator()); }

            match *block_data.terminator() {
                mir::Terminator::Return => break,
                mir::Terminator::Goto { target } => block = target,

                // mir::Terminator::Call { ref func, ref args, ref destination, .. } => {
                //     let ptr = destination.as_ref().map(|&(ref lv, _)| self.lvalue_to_ptr(lv));
                //     let func_val = self.operand_to_ptr(func);

                //     if let Value::Func(def_id) = func_val {
                //         let mir_data;
                //         let mir = match self.tcx.map.as_local_node_id(def_id) {
                //             Some(node_id) => self.mir_map.map.get(&node_id).unwrap(),
                //             None => {
                //                 let cstore = &self.tcx.sess.cstore;
                //                 mir_data = cstore.maybe_get_item_mir(self.tcx, def_id).unwrap();
                //                 &mir_data
                //             }
                //         };

                //         let arg_vals: Vec<Value> =
                //             args.iter().map(|arg| self.operand_to_ptr(arg)).collect();

                //         self.call(mir, &arg_vals, ptr);

                //         if let Some((_, target)) = *destination {
                //             block = target;
                //         }
                //     } else {
                //         panic!("tried to call a non-function value: {:?}", func_val);
                //     }
                // }

                // mir::Terminator::If { ref cond, targets: (then_target, else_target) } => {
                //     match self.operand_to_ptr(cond) {
                //         Value::Bool(true) => block = then_target,
                //         Value::Bool(false) => block = else_target,
                //         cond_val => panic!("Non-boolean `if` condition value: {:?}", cond_val),
                //     }
                // }

                // mir::Terminator::SwitchInt { ref discr, ref values, ref targets, .. } => {
                //     let discr_val = self.read_lvalue(discr);

                //     let index = values.iter().position(|v| discr_val == self.const_to_ptr(v))
                //         .expect("discriminant matched no values");

                //     block = targets[index];
                // }

                // mir::Terminator::Switch { ref discr, ref targets, .. } => {
                //     let discr_val = self.read_lvalue(discr);

                //     if let Value::Adt { variant, .. } = discr_val {
                //         block = targets[variant];
                //     } else {
                //         panic!("Switch on non-Adt value: {:?}", discr_val);
                //     }
                // }

                mir::Terminator::Drop { target, .. } => {
                    // TODO: Handle destructors and dynamic drop.
                    block = target;
                }

                mir::Terminator::Resume => unimplemented!(),
                _ => unimplemented!(),
            }
        }

        self.pop_stack_frame();
        Ok(())
    }

    fn lvalue_to_ptr(&self, lvalue: &mir::Lvalue) -> EvalResult<Pointer> {
        let frame = self.stack.last().expect("no call frames exists");

        let ptr = match *lvalue {
            mir::Lvalue::ReturnPointer => frame.return_ptr.clone()
                .expect("ReturnPointer used in a function with no return value"),
            mir::Lvalue::Arg(i)  => frame.arg_ptr(i),
            mir::Lvalue::Var(i)  => frame.var_ptr(i),
            mir::Lvalue::Temp(i) => frame.temp_ptr(i),
            ref l => panic!("can't handle lvalue: {:?}", l),
        };

        Ok(ptr)

        //     mir::Lvalue::Projection(ref proj) => {
        //         let base_ptr = self.lvalue_to_ptr(&proj.base);

        //         match proj.elem {
        //             mir::ProjectionElem::Field(field, _) => {
        //                 base_ptr.offset(field.index())
        //             }

        //             mir::ProjectionElem::Downcast(_, variant) => {
        //                 let adt_val = self.read_pointer(base_ptr);
        //                 if let Value::Adt { variant: actual_variant, data_ptr } = adt_val {
        //                     debug_assert_eq!(variant, actual_variant);
        //                     data_ptr
        //                 } else {
        //                     panic!("Downcast attempted on non-ADT: {:?}", adt_val)
        //                 }
        //             }

        //             mir::ProjectionElem::Deref => {
        //                 let ptr_val = self.read_pointer(base_ptr);
        //                 if let Value::Pointer(ptr) = ptr_val {
        //                     ptr
        //                 } else {
        //                     panic!("Deref attempted on non-pointer: {:?}", ptr_val)
        //                 }
        //             }

        //             mir::ProjectionElem::Index(ref _operand) => unimplemented!(),
        //             mir::ProjectionElem::ConstantIndex { .. } => unimplemented!(),
        //         }
        //     }

        //     _ => unimplemented!(),
        // }
    }

    fn eval_binary_op(&mut self, bin_op: mir::BinOp, left: Pointer, right: Pointer, dest: &Pointer)
            -> EvalResult<()> {
        match (&left.repr, &right.repr, &dest.repr) {
            (&Repr::Int, &Repr::Int, &Repr::Int) => {
                let l = try!(self.memory.read_int(&left));
                let r = try!(self.memory.read_int(&right));
                let n = match bin_op {
                    mir::BinOp::Add    => l + r,
                    mir::BinOp::Sub    => l - r,
                    mir::BinOp::Mul    => l * r,
                    mir::BinOp::Div    => l / r,
                    mir::BinOp::Rem    => l % r,
                    mir::BinOp::BitXor => l ^ r,
                    mir::BinOp::BitAnd => l & r,
                    mir::BinOp::BitOr  => l | r,
                    mir::BinOp::Shl    => l << r,
                    mir::BinOp::Shr    => l >> r,
                    _                  => unimplemented!(),
                    // mir::BinOp::Eq     => Value::Bool(l == r),
                    // mir::BinOp::Lt     => Value::Bool(l < r),
                    // mir::BinOp::Le     => Value::Bool(l <= r),
                    // mir::BinOp::Ne     => Value::Bool(l != r),
                    // mir::BinOp::Ge     => Value::Bool(l >= r),
                    // mir::BinOp::Gt     => Value::Bool(l > r),
                };
                self.memory.write_int(dest, n)
            }
            (l, r, o) =>
                panic!("unhandled binary operation: {:?}({:?}, {:?}) into {:?}", bin_op, l, r, o),
        }
    }

    fn eval_rvalue_into(&mut self, rvalue: &mir::Rvalue, dest: &Pointer) -> EvalResult<()> {
        match *rvalue {
            mir::Rvalue::Use(ref operand) => {
                let src = try!(self.operand_to_ptr(operand));
                try!(self.memory.copy(&src, dest, src.repr.size()));
            }

            mir::Rvalue::BinaryOp(bin_op, ref left, ref right) => {
                let left_ptr = try!(self.operand_to_ptr(left));
                let right_ptr = try!(self.operand_to_ptr(right));
                try!(self.eval_binary_op(bin_op, left_ptr, right_ptr, dest));
            }

            mir::Rvalue::UnaryOp(un_op, ref operand) => {
                let ptr = try!(self.operand_to_ptr(operand));
                let m = try!(self.memory.read_int(&ptr));
                let n = match (un_op, ptr.repr) {
                    (mir::UnOp::Not, Repr::Int) => !m,
                    (mir::UnOp::Neg, Repr::Int) => -m,
                    (_, ref p) => panic!("unhandled binary operation: {:?}({:?})", un_op, p),
                };
                try!(self.memory.write_int(dest, n));
            }

            mir::Rvalue::Aggregate(mir::AggregateKind::Tuple, ref operands) => {
                match dest.repr {
                    Repr::Aggregate { ref fields, .. } => {
                        for (field, operand) in fields.iter().zip(operands) {
                            let src = try!(self.operand_to_ptr(operand));
                            try!(self.memory.copy(&src, &dest.offset(field.offset), src.repr.size()));
                        }
                    }

                    _ => panic!("attempted to write tuple rvalue '{:?}' into non-aggregate pointer '{:?}'",
                                rvalue, dest)
                }
            }

            // mir::Rvalue::Ref(_region, _kind, ref lvalue) => {
            //     Value::Pointer(self.lvalue_to_ptr(lvalue))
            // }

            // mir::Rvalue::Aggregate(mir::AggregateKind::Adt(ref adt_def, variant, _substs),
            //                        ref operands) => {
            //     let max_fields = adt_def.variants
            //         .iter()
            //         .map(|v| v.fields.len())
            //         .max()
            //         .unwrap_or(0);

            //     let ptr = self.allocate_aggregate(max_fields);

            //     for (i, operand) in operands.iter().enumerate() {
            //         let val = self.operand_to_ptr(operand);
            //         self.write_pointer(ptr.offset(i), val);
            //     }

            //     Value::Adt { variant: variant, data_ptr: ptr }
            // }

            ref r => panic!("can't handle rvalue: {:?}", r),
        }

        Ok(())
    }

    fn operand_to_ptr(&mut self, op: &mir::Operand) -> EvalResult<Pointer> {
        match *op {
            mir::Operand::Consume(ref lvalue) => self.lvalue_to_ptr(lvalue),

            mir::Operand::Constant(ref constant) => {
                match constant.literal {
                    mir::Literal::Value { ref value } => self.const_to_ptr(value),

                    mir::Literal::Item { def_id, kind, .. } => match kind {
                        // mir::ItemKind::Function | mir::ItemKind::Method => Value::Func(def_id),
                        _ => panic!("can't handle item literal: {:?}", constant.literal),
                    },
                }
            }
        }
    }

    fn const_to_ptr(&mut self, const_val: &const_eval::ConstVal) -> EvalResult<Pointer> {
        match *const_val {
            const_eval::ConstVal::Float(_f)         => unimplemented!(),
            const_eval::ConstVal::Int(n) => {
                let ptr = self.memory.allocate(Repr::Int);
                try!(self.memory.write_int(&ptr, n));
                Ok(ptr)
            }
            const_eval::ConstVal::Uint(_u)          => unimplemented!(),
            const_eval::ConstVal::Str(ref _s)       => unimplemented!(),
            const_eval::ConstVal::ByteStr(ref _bs)  => unimplemented!(),
            const_eval::ConstVal::Bool(b)           => unimplemented!(),
            const_eval::ConstVal::Struct(_node_id)  => unimplemented!(),
            const_eval::ConstVal::Tuple(_node_id)   => unimplemented!(),
            const_eval::ConstVal::Function(_def_id) => unimplemented!(),
            const_eval::ConstVal::Array(_, _)       => unimplemented!(),
            const_eval::ConstVal::Repeat(_, _)      => unimplemented!(),
        }
    }
}

pub fn interpret_start_points<'tcx>(tcx: &TyCtxt<'tcx>, mir_map: &MirMap<'tcx>) {
    for (&id, mir) in &mir_map.map {
        for attr in tcx.map.attrs(id) {
            if attr.check_name("miri_run") {
                let item = tcx.map.expect_item(id);

                println!("Interpreting: {}", item.name);

                let mut miri = Interpreter::new(tcx, mir_map);
                let return_ptr = match mir.return_ty {
                    ty::FnConverging(ty) => Some(miri.memory.allocate(Repr::from_ty(ty))),
                    ty::FnDiverging => None,
                };
                miri.call(mir, &[], return_ptr.clone()).unwrap();

                if let Some(ret) = return_ptr {
                    println!("Returned: {:?}\n", miri.memory.get(ret.alloc_id).unwrap());
                }
            }
        }
    }
}

fn check_expected(actual: &str, attr: &Attribute) -> bool {
    if let Some(meta_items) = attr.meta_item_list() {
        for meta_item in meta_items {
            if meta_item.check_name("expected") {
                let expected = meta_item.value_str().unwrap();

                if actual == &expected[..] {
                    println!("Test passed!\n");
                } else {
                    println!("Actual value:\t{}\nExpected value:\t{}\n", actual, expected);
                }

                return true;
            }
        }
    }

    false
}
