//@ check-pass
//@ run-rustfix

#![warn(unused)]
#![allow(dead_code)]

struct Foo {
    r#move: u32
}

fn main() {
    let y = Foo { r#move: 3 };

    let _ = match y {
         Foo { r#move } => 0 //~ WARNING unused variable: `r#move`
                             //~| HELP try ignoring the field
                             //~| SUGGESTION r#move: _
    };
}
