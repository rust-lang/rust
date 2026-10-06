//! Tests that GVN does not reuse dereferences of inactive union fields across intervening writes.
//! See <https://github.com/rust-lang/rust/issues/163825>.

//@ test-mir-pass: GVN
//@ compile-flags: -O

#[derive(Clone, Copy)]
union U {
    r: &'static u8,
    p: *mut u8,
}

// EMIT_MIR gvn_union_deref.demo.GVN.diff
#[inline(never)]
fn demo(p: *mut u8) -> (u8, u8) {
    // CHECK-LABEL: fn demo(
    // CHECK: (*_1) = const 2_u8;
    // CHECK: _5 = copy (*{{.*}});
    // CHECK: _0 = (copy _4, copy _5);
    let u = U { p };
    unsafe {
        let a = *u.r;
        *p = 2;
        let b = *u.r;
        (a, b)
    }
}

// Ensure that when there is NO intervening write, GVN still optimizes redundant loads.
// EMIT_MIR gvn_union_deref.no_intervening_write.GVN.diff
#[inline(never)]
fn no_intervening_write(r: &u8) -> (u8, u8) {
    // CHECK-LABEL: fn no_intervening_write(
    // CHECK: _2 = copy (*_1);
    // CHECK: _0 = (copy _2, copy _2);
    let a = *r;
    let b = *r;
    (a, b)
}

fn main() {
    let mut x = 1u8;
    assert_eq!(demo(&raw mut x), (1, 2));
    assert_eq!(no_intervening_write(&10), (10, 10));
}
