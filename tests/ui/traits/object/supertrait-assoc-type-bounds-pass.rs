//@ check-pass
// Regression test for #152607
// Tests that associated type bounds from supertraits are properly checked
// when forming trait objects, and that valid cases compile successfully.

trait Super {
    type Assoc;
}

trait Sub: Super<Assoc: Copy> {}

fn checked_copy<T: Sub<Assoc = i32> + ?Sized>(x: &i32) -> i32 {
    *x
}

fn main() {
    let x: i32 = 42;
    // This should work because i32: Copy
    let _y: i32 = checked_copy::<dyn Sub<Assoc = i32>>(&x);
}
