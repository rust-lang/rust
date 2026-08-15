//@ edition: 2024
//@ compile-flags: -Znext-solver -Zdxf
//@ check-pass

// We should play nice with universes in static coroutines.

#![feature(coroutines, stmt_expr_attributes)]
#![allow(dead_code)]

fn assert_send<T: Send>(_: T) {}

// === U1: Static coroutine with bound regions across yield ===
fn case_u1_coroutine_with_ref() {
    assert_send(#[coroutine] static |_: ()| {
        let x = 42i32;
        let r = &x;
        yield;
        let _ = *r;
    });
}

// === U2: Nested static coroutine ===
fn case_u2_nested_coroutine() {
    assert_send(#[coroutine] static |_: ()| {
        let x = 42i32;
        let r = &x;
        let inner = #[coroutine] static |_: ()| {
            let y = 100i32;
            let s = &y;
            yield;
            let _ = *s;
        };
        yield;
        let _ = (*r, inner);
    });
}

// === U3: Triple-nested static coroutine ===
fn case_u3_triple_nested() {
    assert_send(#[coroutine] static |_: ()| {
        let a = 1i32;
        let ra = &a;
        let mid = #[coroutine] static |_: ()| {
            let b = 2i32;
            let rb = &b;
            let deep = #[coroutine] static |_: ()| {
                let c = 3i32;
                let rc = &c;
                yield;
                let _ = *rc;
            };
            yield;
            let _ = (*rb, deep);
        };
        yield;
        let _ = (*ra, mid);
    });
}

fn main() {
    case_u1_coroutine_with_ref();
    case_u2_nested_coroutine();
    case_u3_triple_nested();
}
