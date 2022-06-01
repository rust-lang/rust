#![allow(dead_code, unused_variables)]

fn main() {}

mod should_lint {
    fn two_helps() {
        let s = &String::new();
        let x: &str = &*s;
    }
}
