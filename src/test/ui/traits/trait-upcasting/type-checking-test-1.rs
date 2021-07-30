#![feature(trait_upcasting)]
#![allow(incomplete_features)]

trait Foo: Bar<i32> + Bar<u32> {}
trait Bar<T> {
    fn bar(&self) -> Option<T> {
        None
    }
}

fn test_specific(x: &dyn Foo) {
    let _ = x as &dyn Bar<i32>; // FIXME: OK, eventually
                                //~^ ERROR non-primitive cast
                                //~^^ ERROR the trait bound `&dyn Foo: Bar<i32>` is not satisfied
    let _ = x as &dyn Bar<u32>; // FIXME: OK, eventually
                                //~^ ERROR non-primitive cast
                                //~^^ ERROR the trait bound `&dyn Foo: Bar<u32>` is not satisfied
}

fn test_unknown_version(x: &dyn Foo) {
    let _ = x as &dyn Bar<_>; // Ambiguous
                              //~^ ERROR non-primitive cast
                              //~^^ ERROR the trait bound `&dyn Foo: Bar<_>` is not satisfied
}

fn test_infer_version(x: &dyn Foo) {
    let a = x as &dyn Bar<_>; // FIXME: OK, eventually
                              //~^ ERROR non-primitive cast
                              //~^^ ERROR the trait bound `&dyn Foo: Bar<u32>` is not satisfied
    let _: Option<u32> = a.bar();
}

fn main() {}
