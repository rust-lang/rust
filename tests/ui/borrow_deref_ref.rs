//@run-rustfix
//@aux-build: proc_macros.rs:proc-macro

#![allow(dead_code, unused_variables)]

extern crate proc_macros;
use proc_macros::with_span;

fn main() {}

mod should_lint {
    fn one_help() {
        let a = &12;
        let b = &*a;

        let b = &mut &*bar(&12);
    }

    fn bar(x: &u32) -> &u32 {
        x
    }
}

// this mod explains why we should not lint `&mut &* (&T)`
mod should_not_lint1 {
    fn foo(x: &mut &u32) {
        *x = &1;
    }

    fn main() {
        let mut x = &0;
        foo(&mut &*x); // should not lint
        assert_eq!(*x, 0);

        foo(&mut x);
        assert_eq!(*x, 1);
    }
}

// similar to should_not_lint1
mod should_not_lint2 {
    struct S<'a> {
        a: &'a u32,
        b: u32,
    }

    fn main() {
        let s = S { a: &1, b: 1 };
        let x = &mut &*s.a;
        *x = &2;
    }
}

with_span!(
    span

    fn just_returning(x: &u32) -> &u32 {
        x
    }

    fn dont_lint_proc_macro() {
        let a = &mut &*just_returning(&12);
    }
);
// this mod explains why we should not lint `& &* (&T)`
mod false_negative {
    fn foo() {
        let x = &12;
        let addr_x = &x as *const _ as usize;
        let addr_y = &&*x as *const _ as usize; // assert ok
        // let addr_y = &x as *const _ as usize; // assert fail
        assert_ne!(addr_x, addr_y);
    }
}
