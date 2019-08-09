// run-pass
#![feature(box_syntax)]

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
    let snoopy = box Dog{name: "snoopy".to_string()};
    let bubbles = box Dog{name: "bubbles".to_string()};
    let barker = [snoopy as Box<dyn Barks>, bubbles as Box<dyn Barks>];

    for pup in &barker {
        println!("{}", pup.bark());
    }
}
