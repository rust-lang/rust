use rustc::front;
use rustc::middle::ty;
use rustc_mir::mir_map::MirMap;
use rustc_mir::repr::{self as mir, Mir};
use syntax::attr::AttrMetaMethods;

#[derive(Clone, Debug)]
enum Value {
    Uninit,
    Int(i64),
}

struct Interpreter<'tcx> {
    mir: &'tcx Mir<'tcx>,
    var_vals: Vec<Value>,
    temp_vals: Vec<Value>,
    result: Value,
}

impl<'tcx> Interpreter<'tcx> {
    fn new(mir: &'tcx Mir<'tcx>) -> Self {
        Interpreter {
            mir: mir,
            var_vals: vec![Value::Uninit; mir.var_decls.len()],
            temp_vals: vec![Value::Uninit; mir.temp_decls.len()],
            result: Value::Uninit,
        }
    }

    fn run(&mut self) {
        let start_block = self.mir.basic_block_data(mir::START_BLOCK);

        for stmt in &start_block.statements {
            use rustc_mir::repr::Lvalue::*;
            use rustc_mir::repr::StatementKind::*;

            println!("  {:?}", stmt);
            match stmt.kind {
                Assign(ref lv, ref rv) => {
                    let val = self.eval_rvalue(rv);

                    let spot = match *lv {
                        Var(i) => &mut self.var_vals[i as usize],
                        Temp(i) => &mut self.temp_vals[i as usize],
                        ReturnPointer => &mut self.result,
                        _ => unimplemented!(),
                    };

                    *spot = val;
                }
                Drop(_kind, ref _lv) => { /* TODO */ },
            }
        }

        println!("  {:?}", start_block.terminator);
        println!("=> {:?}", self.result);
    }

    fn eval_rvalue(&mut self, rv: &mir::Rvalue) -> Value {
        use rustc_mir::repr::Rvalue::*;

        match *rv {
            Use(ref op) => self.eval_operand(op),
            BinaryOp(mir::BinOp::Add, ref left, ref right) => {
                let left_val = self.eval_operand(left);
                let right_val = self.eval_operand(right);
                match (left_val, right_val) {
                    (Value::Int(l), Value::Int(r)) => Value::Int(l + r),
                    _ => unimplemented!(),
                }
            }
            _ => unimplemented!(),
        }
    }

    fn eval_operand(&mut self, op: &mir::Operand) -> Value {
        use rustc::middle::const_eval::ConstVal::*;
        use rustc_mir::repr::Lvalue::*;
        use rustc_mir::repr::Operand::*;

        match *op {
            Consume(Var(i)) => self.var_vals[i as usize].clone(),
            Consume(Temp(i)) => self.temp_vals[i as usize].clone(),
            Constant(ref constant) => {
                match constant.literal {
                    mir::Literal::Value { value: Int(n) } => Value::Int(n),
                    _ => unimplemented!(),
                }
            }
            _ => unimplemented!(),
        }
    }
}

pub fn interpret_start_points<'tcx>(tcx: &ty::ctxt<'tcx>, mir_map: &MirMap<'tcx>) {
    for (&id, mir) in mir_map {
        for attr in tcx.map.attrs(id) {
            if attr.check_name("miri_run") {
                let item = match tcx.map.get(id) {
                    front::map::NodeItem(item) => item,
                    _ => panic!(),
                };
                println!("Interpreting: {}", item.name);
                let mut interpreter = Interpreter::new(mir);
                interpreter.run();
            }
        }
    }
}
