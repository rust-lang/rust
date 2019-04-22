fn id<T>(x: T) -> T { x }

const FOO: usize = 3;

fn foo() -> &'static usize { &id(FOO) }
//~^ ERROR: cannot return reference to temporary value

fn main() {
}
