// We currently accept conflicting associated type bounds with different generics,
// which results in an ICE, since those generics can be instantiated with the
// same concrete type.
// See https://github.com/rust-lang/rust/issues/154662

trait Super<T> {
    type Assoc;
}

trait Sub<T, U>: Super<T, Assoc = u32> + Super<U, Assoc = u64> {
    fn method(&self) {}
}

fn foo<T, U>(x: Option<&dyn Sub<T, U>>) {
    //~^ ERROR the trait `Sub` is not dyn compatible
    if false {
        x.unwrap().method();
        //~^ ERROR the trait `Sub` is not dyn compatible
    }
}

fn main() {
    // This ends up proving that `dyn Sub<i16, i16>` implements `Super<i16>`.
    // However, `dyn Sub<i16, i16>` has bounds for both `Assoc = u32` and `Assoc = u64`,
    // which is nonsense.
    foo::<i16, i16>(None);
    //~^ ERROR the trait `Sub` is not dyn compatible
}
