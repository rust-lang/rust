#![warn(clippy::bufreader_stdin)]

use std::io::{self, BufReader};

macro_rules! stdin_macro {
    () => {
        io::stdin()
    };
}

fn main() {
    let reader = BufReader::new(stdin_macro!());
    //~^ bufreader_stdin
}
