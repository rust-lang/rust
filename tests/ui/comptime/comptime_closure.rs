#![feature(stmt_expr_attributes, comptime)]

const _: () = {
    let f = #[comptime]
    //~^ ERROR: only functions, trait impls, and methods may be comptime
    || ();

    // FIXME(comptime): closures should work, too.
    f();
    //~^ ERROR: cannot call non-const closure in constants
};

fn main() {}
