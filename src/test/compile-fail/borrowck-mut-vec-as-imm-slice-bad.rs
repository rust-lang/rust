fn want_slice(v: &[int]) -> int {
    let mut sum = 0;
    for vec::each_ref(v) |i| { sum += *i; }
    return sum;
}

fn has_mut_vec(+v: @~[mut int]) -> int {
    want_slice(*v) //~ ERROR illegal borrow unless pure
        //~^ NOTE impure due to access to impure function
}

fn main() {
    assert has_mut_vec(@~[mut 1, 2, 3]) == 6;
}