fn f1<'a>(x: u8, y: &'a ...) {}
//~^ ERROR C-variadic type `...` may not be nested inside another type

fn f2<'a>(x: u8, y: Vec<&'a ...>) {}
//~^ ERROR C-variadic type `...` may not be nested inside another type

// Regression test for issue #125847.
fn f3() where for<> ...: {}
//~^ ERROR C-variadic type `...` may not be nested inside another type

fn main() {
    let _recovery_witness: () = 0;
    //~^ ERROR: mismatched types
}
