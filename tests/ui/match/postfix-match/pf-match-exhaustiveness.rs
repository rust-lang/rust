#![feature(postfix_match)]

fn main() {
    Some(1).match { //~ non-exhaustive patterns
        None => {},
    }
}
