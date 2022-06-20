use crate::z3_builder::Z3Builder;
use tracing::debug;
use z3::{Config, Context, Solver};

pub fn example_sat_z3() -> () {
    // Initialize the Z3 and Builder objects
    let cfg = Config::new();
    let ctx = Context::new(&cfg);
    let solver = Solver::new(&ctx);
    let z3_builder = Z3Builder::new(&solver);

    // Create three logical expressions we want to create a conjunction with
    let xand1 = z3_builder.gen_eq("x1", "x2");
    z3_builder.gen_const_int("x_const", 5);
    let xand2 = z3_builder.gen_eq("x1", "x_const");
    let xge1 = z3_builder.gen_ge("x1", "x2");

    // Insert expressions into the builder
    z3_builder.add_assertion(&xand1);
    z3_builder.add_assertion(&xge1);
    z3_builder.add_assertion(&xand2);

    // Attempt resolving the model (and obtaining the respective values of x1, x2)
    debug!("Resolved value: {:?}", z3_builder.check_solver());
    debug!(
        "x1: {:?}; x2: {:?}; x_const: {:?}",
        z3_builder.resolve_variable("x1"),
        z3_builder.resolve_variable("x2"),
        z3_builder.resolve_variable("x_const")
    );
}

pub fn example_unsat_z3() -> () {
    // Initialize the Z3 and Builder objects
    let cfg = Config::new();
    let ctx = Context::new(&cfg);
    let solver = Solver::new(&ctx);
    let z3_builder = Z3Builder::new(&solver);

    // Create three logical expressions we want to create a conjunction with
    let xand1 = z3_builder.gen_eq("x1", "x2");
    z3_builder.gen_const_int("x_const", 5);
    let xand2 = z3_builder.gen_eq("x1", "x_const");
    let xge1 = z3_builder.gen_gt("x1", "x2");

    // Insert expressions into the builder
    z3_builder.add_assertion(&xand1);
    z3_builder.add_assertion(&xge1);
    z3_builder.add_assertion(&xand2);

    // Attempt resolving the model (and obtaining the respective values of x1, x2)
    debug!("Resolved value: {:?}", z3_builder.check_solver());
    debug!(
        "x1: {:?}; x2: {:?}; x_const: {:?}",
        z3_builder.resolve_variable("x1"),
        z3_builder.resolve_variable("x2"),
        z3_builder.resolve_variable("x_const")
    );
}
