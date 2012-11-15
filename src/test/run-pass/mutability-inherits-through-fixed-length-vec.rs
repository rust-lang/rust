fn test1() {
    let mut ints = [0, ..32];
    ints[0] += 1;
    assert ints[0] == 1;
}

fn test2() {
    let mut ints = [0, ..32];
    for vec::each_mut(ints) |i| { *i += 22; }
    for ints.each |i| { assert *i == 22; }
}

fn main() {
    test1();
    test2();
}
