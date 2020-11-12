// compile-flags: -O -Zunsound-mir-opts
// EMIT_MIR unneeded_deref.simple_opt.UnneededDeref.diff
fn simple_opt() -> u64 {
    let x = 5;
    let y = &x;
    let z = *y;
    z
}

// EMIT_MIR unneeded_deref.deep_opt.UnneededDeref.diff
fn deep_opt() -> (u64, u64, u64) {
    let x1 = 1;
    let x2 = 2;
    let x3 = 3;
    let y1 = &x1;
    let y2 = &x2;
    let y3 = &x3;
    let z1 = *y1;
    let z2 = *y2;
    let z3 = *y3;
    (z1, z2, z3)
}

struct S {
    a: u64,
    b: u64,
}

// EMIT_MIR unneeded_deref.opt_struct.UnneededDeref.diff
fn opt_struct(s: S) -> u64 {
    let a = &s.a;
    let b = &s.b;
    let x = *a;
    *b + x
}

// EMIT_MIR unneeded_deref.very_deep_opt.UnneededDeref.diff
fn very_deep_opt() -> (u64, u64, u64, u64, u64, u64, u64, u64, u64) {
    let x1 = 1;
    let x2 = 2;
    let x3 = 3;
    let x4 = 4;
    let x5 = 5;
    let x6 = 6;
    let x7 = 7;
    let x8 = 8;
    let x9 = 9;
    let y1 = &x1;
    let y2 = &x2;
    let y3 = &x3;
    let y4 = &x4;
    let y5 = &x5;
    let y6 = &x6;
    let y7 = &x7;
    let y8 = &x8;
    let y9 = &x9;
    let z1 = *y1;
    let z2 = *y2;
    let z3 = *y3;
    let z4 = *y4;
    let z5 = *y5;
    let z6 = *y6;
    let z7 = *y7;
    let z8 = *y8;
    let z9 = *y9;
    (z1, z2, z3, z4, z5, z6, z7, z8, z9)
}

// EMIT_MIR unneeded_deref.opt_different_bbs.UnneededDeref.diff
fn opt_different_bbs(input: bool) -> u64 {
    let x = 5;
    let y = &x;
    let _ = if input { 2 } else { 3 };
    let z = *y;
    z
}

// EMIT_MIR unneeded_deref.opt_different_bbs2.UnneededDeref.diff
fn opt_different_bbs2(input: bool) -> u64 {
    let x = 5;
    let y = &x;
    let z = if input { *y } else { 3 };
    z
}

// EMIT_MIR unneeded_deref.operand_opt.UnneededDeref.diff
fn operand_opt<T: Copy>(input: Option<T>) -> bool {
    let x = input.is_some();
    let y = input.is_some();
    x == y
}

fn main() {
    simple_opt();
    deep_opt();
    opt_struct(S { a: 0, b: 1 });
    very_deep_opt();
    opt_different_bbs(false);
    opt_different_bbs2(false);
    operand_opt(Some(true));
}
