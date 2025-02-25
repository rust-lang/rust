// Ensure that we properly ignore the `B<Self>` associated type bound on `A::T`
// since that associated type requires `Self: Sized`.

//@ check-pass

struct X(&'static dyn A);

trait A {
    type T: B<Self> where Self: Sized;
}

trait B<T> {}

fn main() {}
