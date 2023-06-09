trait Trait {
    fn bar<'a,'b:'a>(x: &'a str, y: &'b str);
    //~^ NOTE lifetimes in impl do not match this method in trait
}

struct Foo;

impl Trait for Foo {
    fn bar<'a,'b>(x: &'a str, y: &'b str) { //~ ERROR E0195
    //~^ NOTE lifetimes do not match method in trait
    }
}

fn main() {
}
