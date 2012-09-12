fn write(v: &[mut int]) {
    v[0] += 1;
}

fn main() {
    let v = ~[1, 2, 3];
    write(v); //~ ERROR illegal borrow
}
