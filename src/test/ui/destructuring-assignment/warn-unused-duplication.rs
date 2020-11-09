// run-pass

#![feature(destructuring_assignment)]

#![warn(unused_assignments)]

fn main() {
    let mut a;
    // Assignment occurs left-to-right.
    // However, we emit warnings when this happens, so it is clear that this is happening.
    (a, a) = (0, 1); //~ WARN value assigned to `a` is never read
    assert_eq!(a, 1);

    // We can't always tell when a variable is being assigned to twice, which is why we don't try
    // to emit an error, which would be fallible.
    let mut x = 1;
    (*foo(&mut x), *foo(&mut x)) = (5, 6);
    assert_eq!(x, 6);
}

fn foo<'a>(x: &'a mut u32) -> &'a mut u32 {
    x
}
