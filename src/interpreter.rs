use rustc::middle::{const_eval, def_id, ty};
use rustc_mir::mir_map::MirMap;
use rustc_mir::repr::{self as mir, Mir};
use syntax::ast::Attribute;
use syntax::attr::AttrMetaMethods;

use std::iter;

const TRACE_EXECUTION: bool = false;

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Uninit,
    Bool(bool),
    Int(i64), // FIXME: Should be bit-width aware.
    Func(def_id::DefId),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Pointer {
    Stack(usize),
    // TODO(tsion): Heap
}

/// A stack frame:
///
/// ```text
/// +-----------------------+
/// | ReturnPointer         | return value
/// + - - - - - - - - - - - +
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
#[derive(Debug)]
struct Frame {
    offset: usize,
    num_args: usize,
    num_vars: usize,
    num_temps: usize,
    // aggregates
}

impl Frame {
    fn size(&self) -> usize {
        1 + self.num_args + self.num_vars + self.num_temps
    }

    fn return_val_offset(&self) -> usize {
        self.offset
    }

    fn arg_offset(&self, i: u32) -> usize {
        self.offset + 1 + i as usize
    }

    fn var_offset(&self, i: u32) -> usize {
        self.offset + 1 + self.num_args + i as usize
    }

    fn temp_offset(&self, i: u32) -> usize {
        self.offset + 1 + self.num_args + self.num_vars + i as usize
    }
}

struct Interpreter<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    mir_map: &'a MirMap<'tcx>,
    value_stack: Vec<Value>,
    call_stack: Vec<Frame>,
}

impl<'a, 'tcx> Interpreter<'a, 'tcx> {
    fn new(tcx: &'a ty::ctxt<'tcx>, mir_map: &'a MirMap<'tcx>) -> Self {
        Interpreter {
            tcx: tcx,
            mir_map: mir_map,
            value_stack: Vec::new(),
            call_stack: Vec::new(),
        }
    }

    fn push_stack_frame(&mut self, mir: &Mir, args: &[Value]) {
        self.call_stack.push(Frame {
            offset: self.value_stack.len(),
            num_args: mir.arg_decls.len(),
            num_vars: mir.var_decls.len(),
            num_temps: mir.temp_decls.len(),
        });

        let frame = self.call_stack.last().unwrap();
        self.value_stack.extend(iter::repeat(Value::Uninit).take(frame.size()));

        for (i, arg) in args.iter().enumerate() {
            self.value_stack[frame.offset + 1 + i] = arg.clone();
        }
    }

    fn pop_stack_frame(&mut self) {
        let frame = self.call_stack.pop().expect("tried to pop stack frame, but there were none");
        self.value_stack.truncate(frame.offset);
    }

    fn call(&mut self, mir: &Mir, args: &[Value]) -> Value {
        self.push_stack_frame(mir, args);
        let mut block = mir::START_BLOCK;

        loop {
            let block_data = mir.basic_block_data(block);

            for stmt in &block_data.statements {
                if TRACE_EXECUTION { println!("{:?}", stmt); }

                match stmt.kind {
                    mir::StatementKind::Assign(ref lvalue, ref rvalue) => {
                        let ptr = self.eval_lvalue(lvalue);
                        let value = self.eval_rvalue(rvalue);
                        self.write_pointer(ptr, value);
                    }

                    mir::StatementKind::Drop(_kind, ref _lv) => {
                        // TODO
                    },
                }
            }

            if TRACE_EXECUTION { println!("{:?}", block_data.terminator); }

            match block_data.terminator {
                mir::Terminator::Return => break,
                mir::Terminator::Goto { target } => block = target,

                mir::Terminator::Call { data: mir::CallData { ref destination, ref func, ref args }, targets } => {
                    let ptr = self.eval_lvalue(destination);
                    let func_val = self.eval_operand(func);

                    if let Value::Func(def_id) = func_val {
                        let node_id = self.tcx.map.as_local_node_id(def_id).unwrap();
                        let mir = &self.mir_map[&node_id];
                        let arg_vals: Vec<Value> =
                            args.iter().map(|arg| self.eval_operand(arg)).collect();

                        // FIXME: Pass the destination lvalue such that the ReturnPointer inside
                        // the function call will point to the destination.
                        let return_val = self.call(mir, &arg_vals);
                        self.write_pointer(ptr, return_val);
                        block = targets[0];
                    } else {
                        panic!("tried to call a non-function value: {:?}", func_val);
                    }
                }

                mir::Terminator::If { ref cond, targets } => {
                    match self.eval_operand(cond) {
                        Value::Bool(true) => block = targets[0],
                        Value::Bool(false) => block = targets[1],
                        cond_val => panic!("Non-boolean `if` condition value: {:?}", cond_val),
                    }
                }

                mir::Terminator::SwitchInt { ref discr, switch_ty: _, ref values, ref targets } => {
                    let discr_val = self.read_lvalue(discr);

                    let index = values.iter().position(|v| discr_val == self.eval_constant(v))
                        .expect("discriminant matched no values");

                    block = targets[index];
                }

                // mir::Terminator::Diverge => unimplemented!(),
                // mir::Terminator::Panic { target } => unimplemented!(),
                // mir::Terminator::Switch { ref discr, adt_def, ref targets } => unimplemented!(),
                _ => unimplemented!(),
            }
        }

        let ret_val = self.read_lvalue(&mir::Lvalue::ReturnPointer);
        self.pop_stack_frame();
        ret_val
    }

    fn eval_lvalue(&self, lvalue: &mir::Lvalue) -> Pointer {
        let frame = self.call_stack.last().expect("missing call frame");

        match *lvalue {
            mir::Lvalue::ReturnPointer => Pointer::Stack(frame.return_val_offset()),
            mir::Lvalue::Arg(i)  => Pointer::Stack(frame.arg_offset(i)),
            mir::Lvalue::Var(i)  => Pointer::Stack(frame.var_offset(i)),
            mir::Lvalue::Temp(i) => Pointer::Stack(frame.temp_offset(i)),
            _ => unimplemented!(),
        }
    }

    fn eval_binary_op(&mut self, bin_op: mir::BinOp, left: Value, right: Value) -> Value {
        match (left, right) {
            (Value::Int(l), Value::Int(r)) => {
                match bin_op {
                    mir::BinOp::Add    => Value::Int(l + r),
                    mir::BinOp::Sub    => Value::Int(l - r),
                    mir::BinOp::Mul    => Value::Int(l * r),
                    mir::BinOp::Div    => Value::Int(l / r),
                    mir::BinOp::Rem    => Value::Int(l % r),
                    mir::BinOp::BitXor => Value::Int(l ^ r),
                    mir::BinOp::BitAnd => Value::Int(l & r),
                    mir::BinOp::BitOr  => Value::Int(l | r),
                    mir::BinOp::Shl    => Value::Int(l << r),
                    mir::BinOp::Shr    => Value::Int(l >> r),
                    mir::BinOp::Eq     => Value::Bool(l == r),
                    mir::BinOp::Lt     => Value::Bool(l < r),
                    mir::BinOp::Le     => Value::Bool(l <= r),
                    mir::BinOp::Ne     => Value::Bool(l != r),
                    mir::BinOp::Ge     => Value::Bool(l >= r),
                    mir::BinOp::Gt     => Value::Bool(l > r),
                }
            }

            _ => unimplemented!(),
        }
    }

    fn eval_rvalue(&mut self, rvalue: &mir::Rvalue) -> Value {
        match *rvalue {
            mir::Rvalue::Use(ref operand) => self.eval_operand(operand),

            mir::Rvalue::BinaryOp(bin_op, ref left, ref right) => {
                let left_val = self.eval_operand(left);
                let right_val = self.eval_operand(right);
                self.eval_binary_op(bin_op, left_val, right_val)
            }

            mir::Rvalue::UnaryOp(un_op, ref operand) => {
                match (un_op, self.eval_operand(operand)) {
                    (mir::UnOp::Not, Value::Int(n)) => Value::Int(!n),
                    (mir::UnOp::Neg, Value::Int(n)) => Value::Int(-n),
                    _ => unimplemented!(),
                }
            }

            // mir::Rvalue::Aggregate(mir::AggregateKind::Adt(ref adt_def, variant, substs),
            //                        ref operands) => {
            //     let num_fields = adt_def.variants[variant].fields.len();
            //     debug_assert_eq!(num_fields, operands.len());

            //     let data = operands.iter().map(|op| self.eval_operand(op)).collect();
            //     Value::Adt(variant, data)
            // }

            _ => unimplemented!(),
        }
    }

    fn eval_operand(&mut self, op: &mir::Operand) -> Value {
        match *op {
            mir::Operand::Consume(ref lvalue) => self.read_lvalue(lvalue),

            mir::Operand::Constant(ref constant) => {
                match constant.literal {
                    mir::Literal::Value { ref value } => self.eval_constant(value),

                    mir::Literal::Item { def_id, substs: _ } => {
                        Value::Func(def_id)
                    }
                }
            }
        }
    }

    fn eval_constant(&self, const_val: &const_eval::ConstVal) -> Value {
        match *const_val {
            const_eval::ConstVal::Float(_f)         => unimplemented!(),
            const_eval::ConstVal::Int(i)            => Value::Int(i),
            const_eval::ConstVal::Uint(_u)          => unimplemented!(),
            const_eval::ConstVal::Str(ref _s)       => unimplemented!(),
            const_eval::ConstVal::ByteStr(ref _bs)  => unimplemented!(),
            const_eval::ConstVal::Bool(b)           => Value::Bool(b),
            const_eval::ConstVal::Struct(_node_id)  => unimplemented!(),
            const_eval::ConstVal::Tuple(_node_id)   => unimplemented!(),
            const_eval::ConstVal::Function(_def_id) => unimplemented!(),
        }
    }

    fn read_lvalue(&self, lvalue: &mir::Lvalue) -> Value {
        self.read_pointer(self.eval_lvalue(lvalue))
    }

    fn read_pointer(&self, p: Pointer) -> Value {
        match p {
            Pointer::Stack(offset) => self.value_stack[offset].clone(),
        }
    }

    fn write_pointer(&mut self, p: Pointer, val: Value) {
        match p {
            Pointer::Stack(offset) => self.value_stack[offset] = val,
        }
    }
}

pub fn interpret_start_points<'tcx>(tcx: &ty::ctxt<'tcx>, mir_map: &MirMap<'tcx>) {
    for (&id, mir) in mir_map {
        for attr in tcx.map.attrs(id) {
            if attr.check_name("miri_run") {
                let item = tcx.map.expect_item(id);

                println!("Interpreting: {}", item.name);
                let mut interpreter = Interpreter::new(tcx, mir_map);
                let val = interpreter.call(mir, &[]);
                let val_str = format!("{:?}", val);

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
