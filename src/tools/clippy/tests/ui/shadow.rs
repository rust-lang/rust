//@aux-build:proc_macro_derive.rs

#![warn(clippy::shadow_same, clippy::shadow_reuse, clippy::shadow_unrelated)]
#![allow(
    clippy::let_unit_value,
    clippy::needless_if,
    clippy::redundant_guards,
    clippy::redundant_locals
)]

extern crate proc_macro_derive;

#[derive(proc_macro_derive::ShadowDerive)]
pub struct Nothing;

macro_rules! reuse {
    ($v:ident) => {
        let $v = $v + 1;
    };
}

fn shadow_same() {
    let x = 1;
    let x = x;
    //~^ shadow_same
    let mut x = &x;
    //~^ shadow_same
    let x = &mut x;
    //~^ shadow_same
    let x = *x;
    //~^ shadow_same
}

fn shadow_reuse() -> Option<()> {
    let x = ([[0]], ());
    let x = x.0;
    //~^ shadow_reuse
    let x = x[0];
    //~^ shadow_reuse
    let [x] = x;
    //~^ shadow_reuse
    let x = Some(x);
    //~^ shadow_reuse
    let x = foo(x);
    //~^ shadow_reuse
    let x = || x;
    //~^ shadow_reuse
    let x = Some(1).map(|_| x)?;
    //~^ shadow_reuse
    let y = 1;
    let y = match y {
        //~^ shadow_reuse
        1 => 2,
        _ => 3,
    };
    None
}

fn shadow_reuse_macro() {
    let x = 1;
    // this should not warn
    reuse!(x);
}

fn shadow_unrelated() {
    let x = 1;
    let x = 2;
    //~^ shadow_unrelated
}

fn syntax() {
    fn f(x: u32) {
        let x = 1;
        //~^ shadow_unrelated
    }
    let x = 1;
    match Some(1) {
        Some(1) => {},
        Some(x) => {
            //~^ shadow_unrelated
            let x = 1;
            //~^ shadow_unrelated
        },
        _ => {},
    }
    if let Some(x) = Some(1) {}
    //~^ shadow_unrelated
    while let Some(x) = Some(1) {}
    //~^ shadow_unrelated
    let _ = |[x]: [u32; 1]| {
        //~^ shadow_unrelated
        let x = 1;
        //~^ shadow_unrelated
    };
    let y = Some(1);
    if let Some(y) = y {}
    //~^ shadow_reuse
}

fn negative() {
    match Some(1) {
        Some(x) if x == 1 => {},
        Some(x) => {},
        None => {},
    }
    match [None, Some(1)] {
        [Some(x), None] | [None, Some(x)] => {},
        _ => {},
    }
    if let Some(x) = Some(1) {
        let y = 1;
    } else {
        let x = 1;
        let y = 1;
    }
    let x = 1;
    #[allow(clippy::shadow_unrelated)]
    let x = 1;
}

fn foo<T>(_: T) {}

fn question_mark() -> Option<()> {
    let val = 1;
    // `?` expands with a `val` binding
    None?;
    None
}

pub async fn foo1(_a: i32) {}

pub async fn foo2(_a: i32, _b: i64) {
    let _b = _a;
    //~^ shadow_unrelated
}

fn ice_8748() {
    let _ = [0; {
        let x = 1;
        if let Some(x) = Some(1) { x } else { 1 }
        //~^ shadow_unrelated
    }];
}

// https://github.com/rust-lang/rust-clippy/issues/10780
fn shadow_closure() {
    // These are not shadow_unrelated; but they are correctly shadow_reuse
    let x = Some(1);
    #[allow(clippy::shadow_reuse)]
    let y = x.map(|x| x + 1);
    let z = x.map(|x| x + 1);
    //~^ shadow_reuse
    let a: Vec<Option<u8>> = [100u8, 120, 140]
        .iter()
        .map(|i| i.checked_mul(2))
        .map(|i| i.map(|i| i - 10))
        //~^ shadow_reuse
        .collect();
}

struct Issue13795 {
    value: i32,
}

fn issue13795(value: Issue13795) {
    let Issue13795 { value, .. } = value;
    //~^ shadow_same
}

fn issue14377() {
    let a;
    let b;
    (a, b) = (0, 1);

    struct S {
        c: i32,
        d: i32,
    }

    let c;
    let d;
    S { c, d } = S { c: 1, d: 2 };
}

fn main() {}
