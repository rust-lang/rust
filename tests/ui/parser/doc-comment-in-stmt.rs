//@ run-rustfix
#![allow(unused)]
fn foo() -> bool {
    false
    //!self.allow_ty_infer()
    //~^ ERROR found doc comment
}

fn bar() -> bool {
    false
    /*! bar */ //~ ERROR found doc comment
}

fn baz() -> i32 {
    1 /** baz */ //~ ERROR found doc comment
}

fn quux() -> i32 {
    2 /// quux
    //~^ ERROR found doc comment
}

fn main() {
    let x = 0;
    let y = x.max(1) //!foo //~ ERROR found doc comment
        .min(2);
}
