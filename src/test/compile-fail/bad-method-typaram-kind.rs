fn foo<T>() {
    1u.bar::<T>(); //~ ERROR: missing `copy`
}

trait bar {
    fn bar<T:copy>();
}

impl uint: bar {
    fn bar<T:copy>() {
    }
}

fn main() {}
