use rustc::middle::{const_eval, ty};
use rustc_mir::mir_map::MirMap;
use rustc_mir::repr::{self as mir, Mir};
use syntax::ast::Attribute;
use syntax::attr::AttrMetaMethods;

use std::iter;

#[derive(Clone, Debug)]
enum Value {
    Uninit,
    Bool(bool),
    Int(i64), // FIXME: Should be bit-width aware.
}

#[derive(Debug)]
struct Frame {
    offset: usize,
    num_args: usize,
    num_vars: usize,
    num_temps: usize,
}

struct Interpreter {
    value_stack: Vec<Value>,
    call_stack: Vec<Frame>,
}

impl Interpreter {
    fn new() -> Self {
        Interpreter {
            value_stack: Vec::new(),
            call_stack: Vec::new(),
        }
    }

    fn push_stack_frame(&mut self, mir: &Mir, _args: &[Value]) {
        self.call_stack.push(Frame {
            offset: self.value_stack.len(),
            num_args: mir.arg_decls.len(),
            num_vars: mir.var_decls.len(),
            num_temps: mir.temp_decls.len(),
        });

        let frame = self.call_stack.last().unwrap();
        let frame_size = 1 + frame.num_args + frame.num_vars + frame.num_temps;
        self.value_stack.extend(iter::repeat(Value::Uninit).take(frame_size));

        // TODO(tsion): Write args into value_stack.
    }

    fn call(&mut self, mir: &Mir, args: &[Value]) -> Value {
        self.push_stack_frame(mir, args);
        let mut block = mir::START_BLOCK;

        loop {
            use rustc_mir::repr::Terminator::*;

            let block_data = mir.basic_block_data(block);

            for stmt in &block_data.statements {
                use rustc_mir::repr::StatementKind::*;

                match stmt.kind {
                    Assign(ref lvalue, ref rvalue) => {
                        let index = self.eval_lvalue(lvalue);
                        let value = self.eval_rvalue(rvalue);
                        self.value_stack[index] = value;
                    }

                    Drop(_kind, ref _lv) => {
                        // TODO
                    },
                }
            }

            println!("{:?}", block_data.terminator);
            match block_data.terminator {
                Goto { target } => block = target,

                Panic { target: _target } => unimplemented!(),

                If { ref cond, targets } => {
                    match self.eval_operand(&cond) {
                        Value::Bool(true) => block = targets[0],
                        Value::Bool(false) => block = targets[1],
                        cond_val => panic!("Non-boolean `if` condition value: {:?}", cond_val),
                    }
                }

                Return => break,

                _ => unimplemented!(),
            }
        }

        self.value_stack[self.eval_lvalue(&mir::Lvalue::ReturnPointer)].clone()
    }

    fn eval_lvalue(&self, lvalue: &mir::Lvalue) -> usize {
        use rustc_mir::repr::Lvalue::*;

        let frame = self.call_stack.last().expect("missing call frame");

        match *lvalue {
            ReturnPointer => frame.offset,
            Arg(i)  => frame.offset + 1 + i as usize,
            Var(i)  => frame.offset + 1 + frame.num_args + i as usize,
            Temp(i) => frame.offset + 1 + frame.num_args + frame.num_vars + i as usize,
            _ => unimplemented!(),
        }
    }

    fn eval_rvalue(&mut self, rvalue: &mir::Rvalue) -> Value {
        use rustc_mir::repr::Rvalue::*;
        use rustc_mir::repr::BinOp::*;
        use rustc_mir::repr::UnOp::*;

        match *rvalue {
            Use(ref operand) => self.eval_operand(operand),

            BinaryOp(bin_op, ref left, ref right) => {
                match (self.eval_operand(left), self.eval_operand(right)) {
                    (Value::Int(l), Value::Int(r)) => {
                        match bin_op {
                            Add => Value::Int(l + r),
                            Sub => Value::Int(l - r),
                            Mul => Value::Int(l * r),
                            Div => Value::Int(l / r),
                            Rem => Value::Int(l % r),
                            BitXor => Value::Int(l ^ r),
                            BitAnd => Value::Int(l & r),
                            BitOr => Value::Int(l | r),
                            Shl => Value::Int(l << r),
                            Shr => Value::Int(l >> r),
                            Eq => Value::Bool(l == r),
                            Lt => Value::Bool(l < r),
                            Le => Value::Bool(l <= r),
                            Ne => Value::Bool(l != r),
                            Ge => Value::Bool(l >= r),
                            Gt => Value::Bool(l > r),
                        }
                    }
                    _ => unimplemented!(),
                }
            }

            UnaryOp(un_op, ref operand) => {
                match (un_op, self.eval_operand(operand)) {
                    (Not, Value::Int(n)) => Value::Int(!n),
                    (Neg, Value::Int(n)) => Value::Int(-n),
                    _ => unimplemented!(),
                }
            }

            _ => unimplemented!(),
        }
    }

    fn eval_operand(&self, op: &mir::Operand) -> Value {
        use rustc_mir::repr::Operand::*;

        match *op {
            Consume(ref lvalue) => self.value_stack[self.eval_lvalue(lvalue)].clone(),

            Constant(ref constant) => {
                match constant.literal {
                    mir::Literal::Value { value: ref const_val } => self.eval_constant(const_val),
                    mir::Literal::Item { .. } => unimplemented!(),
                }
            }
        }
    }

    fn eval_constant(&self, const_val: &const_eval::ConstVal) -> Value {
        use rustc::middle::const_eval::ConstVal::*;

        match *const_val {
            Float(_f) => unimplemented!(),
            Int(i) => Value::Int(i),
            Uint(_u) => unimplemented!(),
            Str(ref _s) => unimplemented!(),
            ByteStr(ref _bs) => unimplemented!(),
            Bool(_b) => unimplemented!(),
            Struct(_node_id) => unimplemented!(),
            Tuple(_node_id) => unimplemented!(),
            Function(_def_id) => unimplemented!(),
        }
    }
}

pub fn interpret_start_points<'tcx>(tcx: &ty::ctxt<'tcx>, mir_map: &MirMap<'tcx>) {
    for (&id, mir) in mir_map {
        for attr in tcx.map.attrs(id) {
            if attr.check_name("miri_run") {
                let item = tcx.map.expect_item(id);

                println!("Interpreting: {}", item.name);
                let mut interpreter = Interpreter::new();
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
