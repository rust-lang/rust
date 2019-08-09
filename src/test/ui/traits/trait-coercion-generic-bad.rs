struct Struct {
    person: &'static str
}

trait Trait<T> {
    fn f(&self, x: T);
}

impl Trait<&'static str> for Struct {
    fn f(&self, x: &'static str) {
        println!("Hello, {}!", x);
    }
}

fn main() {
    let s: Box<dyn Trait<isize>> = Box::new(Struct { person: "Fred" });
    //~^ ERROR `Struct: Trait<isize>` is not satisfied
    s.f(1);
}
