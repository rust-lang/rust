struct Foo {
    x: &bool,
    //~^ ERROR E0106
}
enum Bar {
    A(u8),
    B(&bool),
   //~^ ERROR E0106
}
type MyStr = &str;
        //~^ ERROR E0106

struct Baz<'a>(&'a str);
struct Buzz<'a, 'b>(&'a str, &'b str);

struct Quux {
    baz: Baz,
    //~^ ERROR E0106
    //~| expected lifetime parameter
    buzz: Buzz,
    //~^ ERROR E0106
    //~| expected 2 lifetime parameters
}

fn main() {
}
