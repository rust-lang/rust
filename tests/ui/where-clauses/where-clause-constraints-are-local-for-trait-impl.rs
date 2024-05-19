fn require_copy<T: Copy>(x: T) {}

struct Bar<T> { x: T }

trait Foo<T> {
    fn needs_copy(self) where T: Copy;
    fn fails_copy(self);
}

// Ensure constraints are only attached to methods locally
impl<T> Foo<T> for Bar<T> {
    fn needs_copy(self) where T: Copy {
        require_copy(self.x);

    }

    fn fails_copy(self) {
        require_copy(self.x);
        //~^ ERROR the trait bound `T: Copy` is not satisfied
    }
}

fn main() {}
