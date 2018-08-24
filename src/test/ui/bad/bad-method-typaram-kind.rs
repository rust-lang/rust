fn foo<T:'static>() {
    1.bar::<T>(); //~ ERROR `T` cannot be sent between threads safely
}

trait bar {
    fn bar<T:Send>(&self);
}

impl bar for usize {
    fn bar<T:Send>(&self) {
    }
}

fn main() {}
