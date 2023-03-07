// run-pass
#![allow(dead_code)]

macro_rules! foo {
    (#[$attr:meta] $x:ident) => {
        #[$attr]
        struct $x {
            x: u32
        }
    }
}

foo! { #[derive(PartialEq, Eq)] Foo }

const FOO: Foo = Foo { x: 0 };

fn main() {
    let y = Foo { x: 1 };
    match y {
        FOO => { }
        _ => { }
    }
}
