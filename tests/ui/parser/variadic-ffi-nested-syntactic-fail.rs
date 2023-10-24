fn f1<'a>(x: u8, y: &'a ...) {}
//~^ ERROR C-variadic type `...` may not be nested inside another type

fn f2<'a>(x: u8, y: Vec<&'a ...>) {}
//~^ ERROR C-variadic type `...` may not be nested inside another type

fn main() {
    // While this is an error, wf-checks happen before typeck, and if any wf-checks
    // encountered errors, we do not continue to typeck, even if the items are
    // unrelated.
    // FIXME(oli-obk): make this report a type mismatch again.
    let _recovery_witness: () = 0;
}
