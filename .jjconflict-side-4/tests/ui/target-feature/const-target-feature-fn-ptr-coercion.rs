//@ only-x86_64
//@ check-pass
//
// Regression test for <https://github.com/rust-lang/rust/issues/152340>.

#![allow(dead_code)]

#[target_feature(enable = "sse2")]
const fn foo() {}

// DefKind::Const
const _: () = unsafe {
    let _: unsafe fn() = foo;
};

// DefKind::AssocConst
struct S;
impl S {
    const C: () = unsafe {
        let _: unsafe fn() = foo;
    };
}

// DefKind::InlineConst
fn bar() {
    let _ = const {
        unsafe {
            let _: unsafe fn() = foo;
        }
    };
}

fn main() {}
