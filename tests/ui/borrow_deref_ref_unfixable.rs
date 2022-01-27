#![allow(dead_code, unused_variables, clippy::explicit_auto_deref)]

fn main() {}

mod should_lint {
    fn two_helps() {
        let s = &String::new();
        let x: &str = &*s;
    }
}
