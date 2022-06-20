use tracing::debug;
use z3::{
    ast::{self, Ast},
    SatResult, Solver,
};

pub struct Z3Builder<'a> {
    i_bool: i32,
    i_int: i32,
    i_const: i32,
    solver: &'a Solver<'a>,
}

impl Z3Builder<'_> {
    pub fn new<'a>(solver: &'a Solver<'a>) -> Z3Builder<'a> {
        Z3Builder { i_bool: 0, i_int: 0, i_const: 0, solver }
    }

    pub fn gen_bool_var<'a>(&'a self, x_name: &'a str) -> ast::Bool<'a> {
        ast::Bool::new_const(self.solver.get_context(), x_name)
    }

    pub fn gen_land<'a>(&'a self, x1_name: &'a str, x2_name: &'a str) -> ast::Bool<'a> {
        let x1 = ast::Bool::new_const(self.solver.get_context(), x1_name);
        let x2 = ast::Bool::new_const(self.solver.get_context(), x2_name);

        // Dummy debug to use i_x struct fields
        debug!("solver: {:?} {:?} {:?}", self.i_bool, self.i_const, self.i_int);

        ast::Bool::and(&self.solver.get_context(), &[&x1, &x2])
    }

    pub fn gen_lor<'a>(&'a self, x1_name: &'a str, x2_name: &'a str) -> ast::Bool<'a> {
        let x1 = ast::Bool::new_const(self.solver.get_context(), x1_name);
        let x2 = ast::Bool::new_const(self.solver.get_context(), x2_name);

        ast::Bool::or(&self.solver.get_context(), &[&x1, &x2])
    }

    pub fn gen_limplies<'a>(&'a self, x1_name: &'a str, x2_name: &'a str) -> ast::Bool<'a> {
        let x1 = ast::Bool::new_const(self.solver.get_context(), x1_name);
        let x2 = ast::Bool::new_const(self.solver.get_context(), x2_name);

        x1.implies(&x2)
    }

    pub fn gen_liff<'a>(&'a self, x1_name: &'a str, x2_name: &'a str) -> ast::Bool<'a> {
        let x1 = ast::Bool::new_const(self.solver.get_context(), x1_name);
        let x2 = ast::Bool::new_const(self.solver.get_context(), x2_name);

        x1.iff(&x2)
    }

    pub fn gen_lnot<'a>(&'a self, x1_name: &'a str) -> ast::Bool<'a> {
        let x1 = ast::Bool::new_const(self.solver.get_context(), x1_name);

        x1.not()
    }

    pub fn gen_gt<'a>(&'a self, x1_name: &'a str, x2_name: &'a str) -> ast::Bool<'a> {
        let x1 = ast::Int::new_const(self.solver.get_context(), x1_name);
        let x2 = ast::Int::new_const(self.solver.get_context(), x2_name);

        x1.gt(&x2)
    }

    pub fn gen_eq<'a>(&'a self, x1_name: &'a str, x2_name: &'a str) -> ast::Bool<'a> {
        let x1 = ast::Int::new_const(self.solver.get_context(), x1_name);
        let x2 = ast::Int::new_const(self.solver.get_context(), x2_name);

        x1._eq(&x2)
    }

    pub fn gen_ge<'a>(&'a self, x1_name: &'a str, x2_name: &'a str) -> ast::Bool<'a> {
        let x1 = ast::Int::new_const(self.solver.get_context(), x1_name);
        let x2 = ast::Int::new_const(self.solver.get_context(), x2_name);

        x1.ge(&x2)
    }

    pub fn gen_mul<'a>(&'a self, x1_name: &'a str, x2_name: &'a str) -> ast::Int<'a> {
        let x1 = ast::Int::new_const(self.solver.get_context(), x1_name);
        let x2 = ast::Int::new_const(self.solver.get_context(), x2_name);

        ast::Int::mul(self.solver.get_context(), &[&x1, &x2])
    }

    pub fn gen_add<'a>(&'a self, x1_name: &'a str, x2_name: &'a str) -> ast::Int<'a> {
        let x1 = ast::Int::new_const(self.solver.get_context(), x1_name);
        let x2 = ast::Int::new_const(self.solver.get_context(), x2_name);

        ast::Int::add(self.solver.get_context(), &[&x1, &x2])
    }

    pub fn gen_const_int<'a>(&'a self, x_name: &'a str, x_int: i32) -> () {
        let x = ast::Int::new_const(self.solver.get_context(), x_name);
        let unnamed_const =
            ast::Int::from_bv(&ast::BV::from_i64(self.solver.get_context(), x_int.into(), 32), true);

        self.add_assertion(&x._eq(&unnamed_const));
    }

    pub fn gen_const_bool<'a>(&'a self, x_name: &'a str, x_bool: bool) -> () {
        let x = ast::Bool::new_const(self.solver.get_context(), x_name);
        let unnamed_const = ast::Bool::from_bool(self.solver.get_context(), x_bool);

        self.add_assertion(&x._eq(&unnamed_const));
    }

    pub fn gen_int<'a>(&'a self, x_name: &'a str, x_int: ast::Int<'a>) -> () {
        let x = ast::Int::new_const(self.solver.get_context(), x_name);

        self.add_assertion(&x._eq(&x_int));
    }

    pub fn gen_bool<'a>(&'a self, x_name: &'a str, x_bool: ast::Bool<'a>) -> () {
        let x = ast::Bool::new_const(self.solver.get_context(), x_name);

        self.add_assertion(&x._eq(&x_bool));
    }

    pub fn check_bounds<'a>(&'a self, x1: &ast::Int<'a>) -> ast::Bool<'a> {
        let min_int = ast::Int::from_bv(
            &ast::BV::from_i64(self.solver.get_context(), i32::MIN.into(), 32),
            true,
        );
        let max_int = ast::Int::from_bv(
            &ast::BV::from_i64(self.solver.get_context(), i32::MAX.into(), 32),
            true,
        );

        ast::Bool::and(self.solver.get_context(), &[&x1.le(&max_int), &x1.ge(&min_int)])
    }

    pub fn create_const_int<'a>(&'a mut self, x: i32) -> String {
        let x1_name = format!("_const_{}", self.i_const);
        self.i_const += 1;
        let unnamed_const =
            ast::Int::from_bv(&ast::BV::from_i64(self.solver.get_context(), x.into(), 32), true);

        let x1_const = ast::Int::new_const(&self.solver.get_context(), x1_name.clone());
        self.add_assertion(&x1_const._eq(&unnamed_const));

        x1_name
    }

    pub fn create_int<'a>(&'a mut self, x: &ast::Int<'a>) -> String {
        let x1_name = format!("_int_{}", self.i_int);
        self.i_int += 1;

        let x1_int = ast::Int::new_const(&self.solver.get_context(), x1_name.clone());
        self.add_assertion(&x1_int._eq(&x));

        x1_name
    }

    pub fn create_bool<'a>(&'a mut self, x: &ast::Bool<'a>) -> String {
        let x1_name = format!("_bool_{}", self.i_bool);
        self.i_bool += 1;

        let x1_bool = ast::Bool::new_const(&self.solver.get_context(), x1_name.clone());
        self.add_assertion(&x1_bool._eq(x));

        x1_name
    }

    pub fn add_assertion<'a>(&'a self, condition: &ast::Bool<'a>) -> () {
        self.solver.assert(condition);
    }

    pub fn resolve_variable<'a>(&'a self, x_name: &'a str) -> Option<i64> {
        let x = ast::Int::new_const(&self.solver.get_context(), x_name);
        if self.check_solver() == SatResult::Sat {
            let model = self.solver.get_model().unwrap();
            Some(model.eval(&x, true).unwrap().as_i64().unwrap())
        } else {
            None
        }
    }

    pub fn check_solver<'a>(&'a self) -> SatResult {
        return self.solver.check();
    }
}
