fn foo<T>() {
    1u.bar::<T>(); //~ ERROR: missing `copy`
}

impl methods for uint {
    fn bar<T:copy>() {
    }
}

fn main() {}
