// revisions: current next
//[next] compile-flags: -Ztrait-solver=next

#![feature(trait_upcasting)]

trait Foo: Bar<i32> + Bar<u32> {}
trait Bar<T> {
    fn bar(&self) -> Option<T> {
        None
    }
}

fn test_specific(x: &dyn Foo) {
    let _ = x as &dyn Bar<i32>; // OK
    let _ = x as &dyn Bar<u32>; // OK
}

fn test_unknown_version(x: &dyn Foo) {
    let _ = x as &dyn Bar<_>; // Ambiguous
                              //~^ ERROR non-primitive cast
}

fn test_infer_version(x: &dyn Foo) {
    let a = x as &dyn Bar<_>; // OK
    let _: Option<u32> = a.bar();
}

fn main() {}
