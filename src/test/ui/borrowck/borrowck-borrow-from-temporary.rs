// Test lifetimes are linked properly when we take reference
// to interior.

fn id<T>(x: T) -> T { x }

struct Foo(isize);

fn foo<'a>() -> &'a isize {
    let &Foo(ref x) = &id(Foo(3)); //~ ERROR borrowed value does not live long enough
    x
}

pub fn main() {
}
