// run-rustfix

#[allow(dead_code, unused_assignments)]
#[warn(clippy::assign_op_pattern)]
fn main() {
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
