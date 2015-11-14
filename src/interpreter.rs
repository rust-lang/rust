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

struct Interpreter {
    stack: Vec<Value>,
    num_vars: usize,
    num_temps: usize,
}

impl Interpreter {
    fn new() -> Self {
        Interpreter {
            stack: Vec::new(),
            num_vars: 0,
            num_temps: 0,
        }
    }

    fn run(&mut self, mir: &Mir) -> Value {
        let start_block = mir.basic_block_data(mir::START_BLOCK);

        self.num_vars = mir.var_decls.len();
        self.num_temps = mir.temp_decls.len();

        self.stack.extend(
            iter::repeat(Value::Uninit).take(1 + self.num_vars + self.num_temps));

        for stmt in &start_block.statements {
            use rustc_mir::repr::StatementKind::*;

            match stmt.kind {
                Assign(ref lvalue, ref rvalue) => {
                    let index = self.eval_lvalue(lvalue);
                    let value = self.eval_rvalue(rvalue);
                    self.stack[index] = value;
                }

                Drop(_kind, ref _lv) => {
                    // TODO
                },
            }
        }

        self.stack[self.eval_lvalue(&mir::Lvalue::ReturnPointer)].clone()
    }

    fn eval_lvalue(&self, lvalue: &mir::Lvalue) -> usize {
        use rustc_mir::repr::Lvalue::*;

        match *lvalue {
            Var(i) => 1 + i as usize,
            Temp(i) => 1 + self.num_vars + i as usize,
            ReturnPointer => 0,
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
            Consume(ref lvalue) => self.stack[self.eval_lvalue(lvalue)].clone(),

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
                let val = interpreter.run(mir);
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
