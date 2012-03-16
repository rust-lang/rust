// error-pattern:instantiating a copyable type parameter with a noncopyable
fn foo<T>() {
    1u.bar::<T>();
}

impl methods for uint {
    fn bar<T:copy>() {
    }
}

fn main() {}
