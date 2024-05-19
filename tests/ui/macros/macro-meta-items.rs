//@ run-pass
//@ compile-flags: --cfg foo --check-cfg=cfg(foo)

#![allow(dead_code)]

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
compiles_fine!(FALSE);
emit!(foo);

fn foo() {
    println!("{}", MISTYPED);
}

pub fn main() {
    // statement
    compiles_fine!(FALSE);
    emit!(FALSE);
    println!("{}", MISTYPED);
}
