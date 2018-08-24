fn require_copy<T: Copy>(x: T) {}

struct Foo<T> { x: T }

// Ensure constraints are only attached to methods locally
impl<T> Foo<T> {
    fn needs_copy(self) where T: Copy {
        require_copy(self.x);

    }

    fn fails_copy(self) {
        require_copy(self.x);
        //~^ ERROR the trait bound `T: std::marker::Copy` is not satisfied
    }
}

fn main() {}
