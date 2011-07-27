// error-pattern:may alias with argument

fn foo(x: &int, f: fn() ) { log x; }

fn whoknows(x: @mutable int) { *x = 10; }

fn main() { let box = @mutable 1; foo(*box, bind whoknows(box)); }