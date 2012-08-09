fn each<T>(x: &[T], op: fn(elem: &T) -> bool) {
    uint::range(0, x.len(), |i| op(&x[i]));
}

fn main() {
    let x = [{mut a: 0}];
    for each(x) |y| {
        let z = &y.a; //~ ERROR illegal borrow unless pure
        x[0].a = 10; //~ NOTE impure due to assigning to mutable field
        log(error, z);
    }
}
