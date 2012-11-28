struct S<T> {
    x: T
}

impl<T> S<T> : core::ops::Drop {
    fn finalize(&self) {
        io::println("bye");
    }
}

fn main() {
    let x = S { x: 1 };
}

