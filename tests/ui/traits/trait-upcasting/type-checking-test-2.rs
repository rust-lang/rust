trait Foo<T>: Bar<i32> + Bar<T> {}
trait Bar<T> {
    fn bar(&self) -> Option<T> {
        None
    }
}

fn test_specific(x: &dyn Foo<i32>) {
    let _ = x as &dyn Bar<i32>; // OK
}

fn test_specific2(x: &dyn Foo<u32>) {
    let _ = x as &dyn Bar<i32>; // OK
}

fn test_specific3(x: &dyn Foo<i32>) {
    let _ = x as &dyn Bar<u32>; // Error
                                //~^ ERROR non-primitive cast
}

fn test_infer_arg(x: &dyn Foo<u32>) {
    let a = x as &dyn Bar<_>; // Ambiguous
                              //~^ ERROR non-primitive cast
    let _ = a.bar();
}

fn main() {}
