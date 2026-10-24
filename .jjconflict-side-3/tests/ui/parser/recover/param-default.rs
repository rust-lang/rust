fn foo(x: i32 = 1) {} //~ ERROR parameter defaults are not supported

type Foo = fn(i32 = 0); //~ ERROR parameter defaults are not supported

fn main() {}
