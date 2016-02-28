// TODO(tsion): Remove this.
#![allow(unused_imports, dead_code, unused_variables)]

use byteorder;
use byteorder::ByteOrder;
use rustc::middle::{const_eval, def_id, ty};
use rustc::middle::cstore::CrateStore;
use rustc::mir::repr::{self as mir, Mir};
use rustc::mir::mir_map::MirMap;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::iter;
use syntax::ast::Attribute;
use syntax::attr::AttrMetaMethods;

const TRACE_EXECUTION: bool = true;

mod memory {
    use byteorder;
    use byteorder::ByteOrder;
    use rustc::middle::ty;
    use std::collections::HashMap;
    use std::mem;

    pub struct Memory {
        next_id: u64,
        alloc_map: HashMap<u64, Value>,
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct AllocId(u64);

    // TODO(tsion): Remove this hack.
    pub fn alloc_id_hack(i: u64) -> AllocId {
        AllocId(i)
    }

    // TODO(tsion): Shouldn't clone values.
    #[derive(Clone, Debug)]
    pub struct Value {
        pub bytes: Vec<u8>,
        // TODO(tsion): relocations
        // TODO(tsion): undef mask
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Pointer {
        pub alloc_id: AllocId,
        pub offset: usize,
        pub repr: Repr,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum Repr {
        Int,

        StackFrame {
            locals: Vec<Repr>,
        }
    }

    impl Memory {
        pub fn new() -> Self {
            Memory { next_id: 0, alloc_map: HashMap::new() }
        }

        pub fn allocate(&mut self, size: usize) -> AllocId {
            let id = AllocId(self.next_id);
            let val = Value { bytes: vec![0; size] };
            self.alloc_map.insert(self.next_id, val);
            self.next_id += 1;
            id
        }

        pub fn allocate_int(&mut self, n: i64) -> AllocId {
            let id = self.allocate(mem::size_of::<i64>());
            byteorder::NativeEndian::write_i64(&mut self.value_mut(id).unwrap().bytes, n);
            id
        }

        pub fn value(&self, id: AllocId) -> Option<&Value> {
            self.alloc_map.get(&id.0)
        }

        pub fn value_mut(&mut self, id: AllocId) -> Option<&mut Value> {
            self.alloc_map.get_mut(&id.0)
        }
    }

    impl Pointer {
        pub fn offset(self, i: usize) -> Self {
            Pointer { offset: self.offset + i, ..self }
        }
    }

    impl Repr {
        // TODO(tsion): Cache these outputs.
        pub fn from_ty(ty: ty::Ty) -> Self {
            match ty.sty {
                ty::TyInt(_) => Repr::Int,
                _ => unimplemented!(),
            }
        }

        pub fn size(&self) -> usize {
            match *self {
                Repr::Int => 8,
                Repr::StackFrame { ref locals } =>
                    locals.iter().map(Repr::size).fold(0, |a, b| a + b)
            }
        }
    }
}
use self::memory::{Pointer, Repr, Value};

#[derive(Clone, Debug)]
pub struct EvalError;

pub type EvalResult<T> = Result<T, EvalError>;

impl Error for EvalError {
    fn description(&self) -> &str {
        "error during MIR evaluation"
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
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

/// A stack frame:
///
/// ```text
/// +-----------------------+
/// | Arg(0)                |
/// | Arg(1)                | arguments
/// | ...                   |
/// | Arg(num_args - 1)     |
/// + - - - - - - - - - - - +
/// | Var(0)                |
/// | Var(1)                | variables
/// | ...                   |
/// | Var(num_vars - 1)     |
/// + - - - - - - - - - - - +
/// | Temp(0)               |
/// | Temp(1)               | temporaries
/// | ...                   |
/// | Temp(num_temps - 1)   |
/// + - - - - - - - - - - - +
/// | Aggregates            | aggregates
/// +-----------------------+
/// ```
// #[derive(Debug)]
// struct Frame {
//     /// A pointer to a stack cell to write the return value of the current call, if it's not a
//     /// divering call.
//     return_ptr: Option<Pointer>,

//     offset: usize,
//     num_args: usize,
//     num_vars: usize,
//     num_temps: usize,
//     num_aggregate_fields: usize,
// }

// impl Frame {
//     fn size(&self) -> usize {
//         self.num_args + self.num_vars + self.num_temps + self.num_aggregate_fields
//     }

//     fn arg_offset(&self, i: usize) -> usize {
//         self.offset + i
//     }

//     fn var_offset(&self, i: usize) -> usize {
//         self.offset + self.num_args + i
//     }

//     fn temp_offset(&self, i: usize) -> usize {
//         self.offset + self.num_args + self.num_vars + i
//     }
// }

struct Interpreter<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    mir_map: &'a MirMap<'tcx>,
    // value_stack: Vec<Value>,
    // call_stack: Vec<Frame>,
    memory: memory::Memory,
}

impl<'a, 'tcx> Interpreter<'a, 'tcx> {
    fn new(tcx: &'a ty::ctxt<'tcx>, mir_map: &'a MirMap<'tcx>) -> Self {
        Interpreter {
            tcx: tcx,
            mir_map: mir_map,
            // value_stack: vec![Value::Uninit], // Allocate a spot for the top-level return value.
            // call_stack: Vec::new(),
            memory: memory::Memory::new(),
        }
    }

    // fn push_stack_frame(&mut self, mir: &Mir, args: &[Value], return_ptr: Option<Pointer>) {
    //     let frame = Frame {
    //         return_ptr: return_ptr,
    //         offset: self.value_stack.len(),
    //         num_args: mir.arg_decls.len(),
    //         num_vars: mir.var_decls.len(),
    //         num_temps: mir.temp_decls.len(),
    //         num_aggregate_fields: 0,
    //     };

    //     self.value_stack.extend(iter::repeat(Value::Uninit).take(frame.size()));

    //     for (i, arg) in args.iter().enumerate() {
    //         self.value_stack[frame.arg_offset(i)] = arg.clone();
    //     }

    //     self.call_stack.push(frame);
    // }

    // fn pop_stack_frame(&mut self) {
    //     let frame = self.call_stack.pop().expect("tried to pop stack frame, but there were none");
    //     self.value_stack.truncate(frame.offset);
    // }

    // fn allocate_aggregate(&mut self, size: usize) -> Pointer {
    //     let frame = self.call_stack.last_mut().expect("missing call frame");
    //     frame.num_aggregate_fields += size;

    //     let ptr = Pointer::Stack(self.value_stack.len());
    //     self.value_stack.extend(iter::repeat(Value::Uninit).take(size));
    //     ptr
    // }

    fn call(&mut self, mir: &Mir, args: &[Value], return_ptr: Option<Pointer>) -> EvalResult<()> {
        // self.push_stack_frame(mir, args, return_ptr);
        let mut block = mir::START_BLOCK;

        loop {
            if TRACE_EXECUTION { println!("Entering block: {:?}", block); }
            let block_data = mir.basic_block_data(block);

            for stmt in &block_data.statements {
                if TRACE_EXECUTION { println!("{:?}", stmt); }

                match stmt.kind {
                    mir::StatementKind::Assign(ref lvalue, ref rvalue) => {
                        let ptr = try!(self.lvalue_to_ptr(lvalue));
                        try!(self.eval_rvalue_into(rvalue, ptr));
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

        // self.pop_stack_frame();

        Ok(())
    }

    fn lvalue_to_ptr(&self, lvalue: &mir::Lvalue) -> EvalResult<Pointer> {
        let ptr = match *lvalue {
            mir::Lvalue::ReturnPointer => Pointer {
                alloc_id: self::memory::alloc_id_hack(0),
                offset: 0,
                repr: Repr::Int,
            },

            _ => unimplemented!(),
        };

        Ok(ptr)

        // let frame = self.call_stack.last().expect("missing call frame");

        // match *lvalue {
        //     mir::Lvalue::ReturnPointer =>
        //         frame.return_ptr.expect("ReturnPointer used in a function with no return value"),
        //     mir::Lvalue::Arg(i)  => Pointer::Stack(frame.arg_offset(i as usize)),
        //     mir::Lvalue::Var(i)  => Pointer::Stack(frame.var_offset(i as usize)),
        //     mir::Lvalue::Temp(i) => Pointer::Stack(frame.temp_offset(i as usize)),

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

    fn eval_binary_op(&mut self, bin_op: mir::BinOp, left: Pointer, right: Pointer, out: Pointer) {
        match (left.repr, right.repr, out.repr) {
            (Repr::Int, Repr::Int, Repr::Int) => {
                let l = byteorder::NativeEndian::read_i64(&self.memory.value(left.alloc_id).unwrap().bytes);
                let r = byteorder::NativeEndian::read_i64(&self.memory.value(right.alloc_id).unwrap().bytes);
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
                byteorder::NativeEndian::write_i64(&mut self.memory.value_mut(out.alloc_id).unwrap().bytes, n);
            }

            _ => unimplemented!(),
        }
    }

    fn eval_rvalue_into(&mut self, rvalue: &mir::Rvalue, out: Pointer) -> EvalResult<()> {
        match *rvalue {
            mir::Rvalue::Use(ref operand) => {
                let ptr = try!(self.operand_to_ptr(operand));
                let val = self.read_pointer(ptr);
                self.write_pointer(out, val);
            }

            mir::Rvalue::BinaryOp(bin_op, ref left, ref right) => {
                let left_ptr = try!(self.operand_to_ptr(left));
                let right_ptr = try!(self.operand_to_ptr(right));
                self.eval_binary_op(bin_op, left_ptr, right_ptr, out)
            }

            mir::Rvalue::UnaryOp(un_op, ref operand) => {
                let ptr = try!(self.operand_to_ptr(operand));
                let m = byteorder::NativeEndian::read_i64(&self.memory.value(ptr.alloc_id).unwrap().bytes);
                let n = match (un_op, ptr.repr) {
                    (mir::UnOp::Not, Repr::Int) => !m,
                    (mir::UnOp::Neg, Repr::Int) => -m,
                    _ => unimplemented!(),
                };
                byteorder::NativeEndian::write_i64(&mut self.memory.value_mut(out.alloc_id).unwrap().bytes, n);
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
                    mir::Literal::Value { ref value } => Ok(self.const_to_ptr(value)),

                    mir::Literal::Item { def_id, kind, .. } => match kind {
                        // mir::ItemKind::Function | mir::ItemKind::Method => Value::Func(def_id),
                        _ => panic!("can't handle item literal: {:?}", constant.literal),
                    },
                }
            }
        }
    }

    fn const_to_ptr(&mut self, const_val: &const_eval::ConstVal) -> Pointer {
        match *const_val {
            const_eval::ConstVal::Float(_f)         => unimplemented!(),
            const_eval::ConstVal::Int(i)            => Pointer {
                alloc_id: self.memory.allocate_int(i),
                offset: 0,
                repr: Repr::Int,
            },
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

    // fn read_lvalue(&self, lvalue: &mir::Lvalue) -> Value {
    //     self.read_pointer(self.lvalue_to_ptr(lvalue))
    // }

    fn read_pointer(&self, p: Pointer) -> Value {
        self.memory.value(p.alloc_id).unwrap().clone()
    }

    fn write_pointer(&mut self, p: Pointer, val: Value) {
        // TODO(tsion): Remove panics.
        let alloc = self.memory.value_mut(p.alloc_id).unwrap();
        for (i, byte) in val.bytes.into_iter().enumerate() {
            alloc.bytes[p.offset + i] = byte;
        }
    }
}

pub fn interpret_start_points<'tcx>(tcx: &ty::ctxt<'tcx>, mir_map: &MirMap<'tcx>) {
    for (&id, mir) in &mir_map.map {
        for attr in tcx.map.attrs(id) {
            if attr.check_name("miri_run") {
                let item = tcx.map.expect_item(id);

                println!("Interpreting: {}", item.name);

                let mut interpreter = Interpreter::new(tcx, mir_map);
                let return_ptr = match mir.return_ty {
                    ty::FnOutput::FnConverging(ty) => {
                        let repr = Repr::from_ty(ty);
                        Some(Pointer {
                            alloc_id: interpreter.memory.allocate(repr.size()),
                            offset: 0,
                            repr: repr,
                        })
                    }
                    ty::FnOutput::FnDiverging => None,
                };
                interpreter.call(mir, &[], return_ptr.clone()).unwrap();

                let val_str = format!("{:?}", interpreter.read_pointer(return_ptr.unwrap()));
                if !check_expected(&val_str, attr) {
                    println!("=> {}\n", val_str);
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
