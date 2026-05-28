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
    let person = "Fred".to_string();
    let person: &str = &person;  //~ ERROR `person` does not live long enough
    let s: Box<dyn Trait<&'static str>> = Box::new(Struct { person: person });
}
