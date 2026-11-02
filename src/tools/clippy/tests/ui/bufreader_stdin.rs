#![warn(clippy::bufreader_stdin)]

use std::io::{self, BufReader};

fn main() {
    let a = io::stdin();
    let reader = BufReader::new(a);
    //~^ bufreader_stdin

    let b = io::stdin().lock();
    let reader = BufReader::new(b);
    //~^ bufreader_stdin
}
