fn foo<T>() {
    1u.bar::<T>(); //~ ERROR: missing `copy`
}

trait bar {
    fn bar<T:Copy>();
}

impl uint: bar {
    fn bar<T:Copy>() {
    }
}

fn main() {}
