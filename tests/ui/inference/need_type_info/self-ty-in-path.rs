// Test that we don't ICE when encountering a `Self` in a path.
struct TestErr<T>(T);

impl<T> TestErr<T> {
    fn func_a<U>() {}

    fn func_b() {
        Self::func_a();
        //~^ ERROR type annotations needed
    }
}

fn main() {}
