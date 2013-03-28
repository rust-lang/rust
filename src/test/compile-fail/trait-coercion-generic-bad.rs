use core::io::println;

struct Struct {
    person: &'static str
}

trait Trait<T> {
    fn f(&self, x: T);
}

impl Trait<&'static str> for Struct {
    fn f(&self, x: &'static str) {
        println(fmt!("Hello, %s!", x));
    }
}

fn main() {
    let s: @Trait<int> = @Struct { person: "Fred" };    //~ ERROR mismatched types
    //~^ ERROR mismatched types
    s.f(1);
}

