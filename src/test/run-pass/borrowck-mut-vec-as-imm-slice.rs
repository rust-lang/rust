fn want_slice(v: &[int]) -> int {
    let mut sum = 0;
    for vec::each(v) { |i| sum += i; }
    ret sum;
}

fn has_mut_vec(+v: ~[mut int]) -> int {
    want_slice(v)
}

fn main() {
    assert has_mut_vec(~[mut 1, 2, 3]) == 6;
}