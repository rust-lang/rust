// Check that E0161 is a hard error in all possible configurations that might
// affect it.

#![crate_type = "lib"]

trait Bar {
    fn f(self);
}

fn foo(x: Box<dyn Bar>) {
    x.f();
    //~^ ERROR E0161
}
