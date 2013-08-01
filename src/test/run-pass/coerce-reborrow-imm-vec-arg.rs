fn sum(x: &[int]) -> int {
    let mut sum = 0;
    foreach y in x.iter() { sum += *y; }
    return sum;
}

fn sum_mut(y: &mut [int]) -> int {
    sum(y)
}

fn sum_imm(y: &[int]) -> int {
    sum(y)
}

/* FIXME #7304
fn sum_const(y: &const [int]) -> int {
    sum(y)
}
*/

pub fn main() {}
