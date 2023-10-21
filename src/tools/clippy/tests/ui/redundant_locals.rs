//@aux-build:proc_macros.rs
#![allow(unused, clippy::no_effect, clippy::needless_pass_by_ref_mut)]
#![warn(clippy::redundant_locals)]

extern crate proc_macros;
use proc_macros::{external, with_span};

fn main() {}

fn immutable() {
    let x = 1;
    let x = x;
}

fn mutable() {
    let mut x = 1;
    let mut x = x;
}

fn upgraded_mutability() {
    let x = 1;
    let mut x = x;
}

fn downgraded_mutability() {
    let mut x = 1;
    let x = x;
}

// see #11290
fn shadow_mutation() {
    let mut x = 1;
    {
        let mut x = x;
        x = 2;
    }
}

fn coercion(par: &mut i32) {
    let par: &i32 = par;

    let x: &mut i32 = &mut 1;
    let x: &i32 = x;
}

fn parameter(x: i32) {
    let x = x;
}

fn many() {
    let x = 1;
    let x = x;
    let x = x;
    let x = x;
    let x = x;
}

fn interleaved() {
    let a = 1;
    let b = 2;
    let a = a;
    let b = b;
}

fn block() {
    {
        let x = 1;
        let x = x;
    }
}

fn closure() {
    || {
        let x = 1;
        let x = x;
    };
    |x: i32| {
        let x = x;
    };
}

fn consequential_drop_order() {
    use std::sync::Mutex;

    let mutex = Mutex::new(1);
    let guard = mutex.lock().unwrap();

    {
        let guard = guard;
    }
}

fn inconsequential_drop_order() {
    let x = 1;

    {
        let x = x;
    }
}

fn macros() {
    macro_rules! rebind {
        ($x:ident) => {
            let $x = 1;
            let $x = $x;
        };
    }

    rebind!(x);

    external! {
        let x = 1;
        let x = x;
    }
    with_span! {
        span
        let x = 1;
        let x = x;
    }

    let x = 10;
    macro_rules! rebind_outer_macro {
        ($x:ident) => {
            let x = x;
        };
    }
    rebind_outer_macro!(y);
}

struct WithDrop(usize);
impl Drop for WithDrop {
    fn drop(&mut self) {}
}

struct InnerDrop(WithDrop);

struct ComposeDrop {
    d: WithDrop,
}

struct WithoutDrop(usize);

fn drop_trait() {
    let a = WithDrop(1);
    let b = WithDrop(2);
    let a = a;
}

fn without_drop() {
    let a = WithoutDrop(1);
    let b = WithoutDrop(2);
    let a = a;
}

fn drop_inner() {
    let a = InnerDrop(WithDrop(1));
    let b = InnerDrop(WithDrop(2));
    let a = a;
}

fn drop_compose() {
    let a = ComposeDrop { d: WithDrop(1) };
    let b = ComposeDrop { d: WithDrop(1) };
    let a = a;
}
