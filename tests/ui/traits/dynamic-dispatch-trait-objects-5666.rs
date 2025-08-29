// https://github.com/rust-lang/rust/issues/5666
//@ run-pass

struct Dog {
    name : String
}

trait Barks {
    fn bark(&self) -> String;
}

impl Barks for Dog {
    fn bark(&self) -> String {
        return format!("woof! (I'm {})", self.name);
    }
}

pub fn main() {
    let snoopy = Box::new(Dog{name: "snoopy".to_string()});
    let bubbles = Box::new(Dog{name: "bubbles".to_string()});
    let barker = [snoopy as Box<dyn Barks>, bubbles as Box<dyn Barks>];

    for pup in &barker {
        println!("{}", pup.bark());
    }
}
