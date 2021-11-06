// run-pass
struct A;

impl A {
    fn take_mutably(&mut self) {}
}

fn identity<T>(t: T) -> T {
    t
}

// Issue 46095
// Built-in indexing should be used even when the index is not
// trivially an integer
// Overloaded indexing would cause wrapped to be borrowed mutably

fn main() {
    let mut a1 = A;
    let mut a2 = A;

    let wrapped = [&mut a1, &mut a2];

    {
        wrapped[0 + 1 - 1].take_mutably();
    }

    {
        wrapped[identity(0)].take_mutably();
    }
}
