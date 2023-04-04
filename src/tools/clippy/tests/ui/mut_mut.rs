// aux-build:proc_macros.rs
#![warn(clippy::mut_mut)]
#![allow(unused)]
#![allow(clippy::no_effect, clippy::uninlined_format_args, clippy::unnecessary_operation)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

fn fun(x: &mut &mut u32) -> bool {
    **x > 0
}

fn less_fun(x: *mut *mut u32) {
    let y = x;
}

macro_rules! mut_ptr {
    ($p:expr) => {
        &mut $p
    };
}

#[allow(unused_mut, unused_variables)]
#[inline_macros]
fn main() {
    let mut x = &mut &mut 1u32;
    {
        let mut y = &mut x;
    }

    if fun(x) {
        let y: &mut &mut u32 = &mut &mut 2;
        **y + **x;
    }

    if fun(x) {
        let y: &mut &mut &mut u32 = &mut &mut &mut 2;
        ***y + **x;
    }

    let mut z = inline!(&mut $(&mut 3u32));
}

fn issue939() {
    let array = [5, 6, 7, 8, 9];
    let mut args = array.iter().skip(2);
    for &arg in &mut args {
        println!("{}", arg);
    }

    let args = &mut args;
    for arg in args {
        println!(":{}", arg);
    }
}

fn issue6922() {
    // do not lint from an external macro
    external!(let mut_mut_ty: &mut &mut u32 = &mut &mut 1u32;);
}

mod issue9035 {
    use std::fmt::Display;

    struct Foo<'a> {
        inner: &'a mut dyn Display,
    }

    impl Foo<'_> {
        fn foo(&mut self) {
            let hlp = &mut self.inner;
            bar(hlp);
        }
    }

    fn bar(_: &mut impl Display) {}
}
