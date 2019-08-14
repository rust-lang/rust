// run-pass
#![allow(dead_code)]
// compile-flags: --cfg foo

macro_rules! compiles_fine {
    ($at:meta) => {
        #[cfg($at)]
        static MISTYPED: () = "foo";
    }
}
macro_rules! emit {
    ($at:meta) => {
        #[cfg($at)]
        static MISTYPED: &'static str = "foo";
    }
}

// item
compiles_fine!(bar);
emit!(foo);

fn foo() {
    println!("{}", MISTYPED);
}

pub fn main() {
    // statement
    compiles_fine!(baz);
    emit!(baz);
    println!("{}", MISTYPED);
}
