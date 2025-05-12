//@ check-pass

// Minimization of issue #59502

trait MyTrait<T> {
    type Output;
}

pub fn pow<T: MyTrait<T, Output = T>>(arg: T) -> T {
    arg
}
