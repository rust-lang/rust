#![feature(rustc_attrs, stmt_expr_attributes)]

const _: () = {
    let f = #[rustc_comptime]
    //~^ ERROR: the `rustc_comptime` attribute cannot be used on closures
    || ();

    // FIXME(comptime): closures should work, too.
    f();
    //~^ ERROR: cannot call non-const closure in constants
};

fn main() {}
