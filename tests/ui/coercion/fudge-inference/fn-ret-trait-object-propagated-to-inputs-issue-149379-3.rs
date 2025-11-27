//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// FIXME(#149379): This should pass, but fails due to fudged expactation
// types which are potentially not well-formed or for whom the function
// where-bounds don't actually hold. And this results in weird bugs when
// later treating these expectations as if they were actually correct..

struct Foo<T> {
    field: T,
    tail: (),
}

struct Bar<T> {
    field: T,
}

fn field_to_box1<T>(x: Foo<T>) -> Box<T> {
    Box::new(x.field)
}

fn field_to_box2<T>(x: &Bar<T>) -> &T {
    &x.field
}

fn field_to_box3<T>(x: &(T,)) -> &T {
    &x.0
}

fn main() {
    let _: Box<dyn Send> = field_to_box1(Foo { field: 1, tail: () });
    //~^ ERROR the size for values of type `dyn Send` cannot be known at compilation time
    //~| ERROR the size for values of type `dyn Send` cannot be known at compilation time
    //~| ERROR mismatched types
    let _: &dyn Send = field_to_box2(&Bar { field: 1 });
    //~^ ERROR the size for values of type `dyn Send` cannot be known at compilation time
    let _: &dyn Send = field_to_box3(&(1,));
    //~^ ERROR mismatched types
}
