fn foo<T>() {
    1u.bar::<T>(); //~ ERROR: missing `copy`
}

trait bar {
    fn bar<T:copy>();
}

impl methods of bar for uint {
    fn bar<T:copy>() {
    }
}

fn main() {}
