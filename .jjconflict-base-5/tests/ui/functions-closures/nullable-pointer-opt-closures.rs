//@ run-pass

use std::mem;

pub fn main() {
    // By Ref Capture
    let a = 10i32;
    let b = Some(|| println!("{}", a));
    // When we capture by reference we can use any of the
    // captures as the discriminant since they're all
    // behind a pointer.
    assert_eq!(mem::size_of_val(&b), mem::size_of::<usize>());

    // By Value Capture
    let a = Box::new(12i32);
    let b = Some(move || println!("{}", a));
    // We captured `a` by value and since it's a `Box` we can use it
    // as the discriminant.
    assert_eq!(mem::size_of_val(&b), mem::size_of::<Box<i32>>());

    // By Value Capture - Transitive case
    let a = "Hello".to_string(); // String -> Vec -> Unique -> NonZero
    let b = Some(move || println!("{}", a));
    // We captured `a` by value and since down the chain it contains
    // a `NonZero` field, we can use it as the discriminant.
    assert_eq!(mem::size_of_val(&b), mem::size_of::<String>());

    // By Value - No Optimization
    let a = 14i32;
    let b = Some(move || println!("{}", a));
    // We captured `a` by value but we can't use it as the discriminant
    // thus we end up with an extra field for the discriminant
    assert_eq!(mem::size_of_val(&b), mem::size_of::<(i32, i32)>());
}
