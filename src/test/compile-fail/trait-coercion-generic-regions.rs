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
    let person = ~"Fred";
    let person: &str = person;  //~ ERROR illegal borrow
    let s: @Trait<&'static str> = @Struct { person: person };
}

