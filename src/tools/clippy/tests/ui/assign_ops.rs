use core::num::Wrapping;

#[allow(dead_code, unused_assignments, clippy::useless_vec)]
#[warn(clippy::assign_op_pattern)]
fn main() {
    let mut a = 5;
    a = a + 1;
    //~^ assign_op_pattern
    a = 1 + a;
    //~^ assign_op_pattern
    a = a - 1;
    //~^ assign_op_pattern
    a = a * 99;
    //~^ assign_op_pattern
    a = 42 * a;
    //~^ assign_op_pattern
    a = a / 2;
    //~^ assign_op_pattern
    a = a % 5;
    //~^ assign_op_pattern
    a = a & 1;
    //~^ assign_op_pattern
    a = 1 - a;
    a = 5 / a;
    a = 42 % a;
    a = 6 << a;
    let mut s = String::new();
    s = s + "bla";
    //~^ assign_op_pattern

    // Issue #9180
    let mut a = Wrapping(0u32);
    a = a + Wrapping(1u32);
    //~^ assign_op_pattern
    let mut v = vec![0u32, 1u32];
    v[0] = v[0] + v[1];
    //~^ assign_op_pattern
    let mut v = vec![Wrapping(0u32), Wrapping(1u32)];
    v[0] = v[0] + v[1];
    let _ = || v[0] = v[0] + v[1];
}
