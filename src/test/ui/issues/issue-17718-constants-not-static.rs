fn id<T>(x: T) -> T { x }

const FOO: usize = 3;

fn foo() -> &'static usize { &id(FOO) }
//~^ ERROR: borrowed value does not live long enough

fn main() {
}
