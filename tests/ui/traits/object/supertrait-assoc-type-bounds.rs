// Regression test for #152607
// Tests that associated type bounds from supertraits are checked
// when forming trait objects.

trait Super {
    type Assoc;
}

trait Sub: Super<Assoc: Copy> {}

fn unchecked_copy<T: Sub<Assoc = String> + ?Sized>(x: &String) -> String {
    *x
}

fn main() {
    let x: String = String::from("abc");
    let _y: String = unchecked_copy::<dyn Sub<Assoc = String>>(&x);
    //~^ ERROR the trait bound `String: Copy` is not satisfied
}
