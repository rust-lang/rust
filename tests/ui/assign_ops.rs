#![feature(plugin)]
#![plugin(clippy)]

#[warn(assign_ops)]
#[allow(unused_assignments)]
fn main() {
    let mut i = 1i32;
    i += 2;
    i += 2 + 17;
    i -= 6;
    i -= 2 - 1;
    i *= 5;
    i *= 1+5;
    i /= 32;
    i /= 32 | 5;
    i /= 32 / 5;
    i %= 42;
    i >>= i;
    i <<= 9 + 6 - 7;
    i += 1 << 5;
}

#[allow(dead_code, unused_assignments)]
#[warn(assign_op_pattern)]
fn bla() {
    let mut a = 5;
    a = a + 1;
    a = 1 + a;
    a = a - 1;
    a = a * 99;
    a = 42 * a;
    a = a / 2;
    a = a % 5;
    a = a & 1;
    a = 1 - a;
    a = 5 / a;
    a = 42 % a;
    a = 6 << a;
    let mut s = String::new();
    s = s + "bla";
}
