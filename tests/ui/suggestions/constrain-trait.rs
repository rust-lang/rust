//@ run-rustfix
// check-only
#![allow(dead_code)]

#[derive(Debug)]
struct Demo {
    a: String
}

trait GetString {
    fn get_a(&self) -> &String;
}

trait UseString: std::fmt::Debug {
    fn use_string(&self) {
        println!("{:?}", self.get_a()); //~ ERROR no method named `get_a` found
    }
}

trait UseString2 {
    fn use_string(&self) {
        println!("{:?}", self.get_a()); //~ ERROR no method named `get_a` found
    }
}

impl GetString for Demo {
    fn get_a(&self) -> &String {
        &self.a
    }
}

impl UseString for Demo {}
impl UseString2 for Demo {}


#[cfg(test)]
mod tests {
    use crate::{Demo, UseString};

    #[test]
    fn it_works() {
        let d = Demo { a: "test".to_string() };
        d.use_string();
    }
}


fn main() {}
