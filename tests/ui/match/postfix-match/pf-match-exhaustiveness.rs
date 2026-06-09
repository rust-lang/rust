#![feature(postfix_match)]

fn main() {
    Some(1).match { //~ ERROR non-exhaustive patterns
        None => {},
    }
}
