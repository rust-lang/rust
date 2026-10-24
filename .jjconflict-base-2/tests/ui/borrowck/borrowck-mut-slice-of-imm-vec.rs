fn write(v: &mut [isize]) {
    v[0] += 1;
}

fn main() {
    let v = vec![1, 2, 3];
    write(&mut v); //~ ERROR cannot borrow
}
