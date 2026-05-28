//@ run-pass

use std::io::{self, Write};

fn f(wr: &mut dyn Write) {
    wr.write_all(b"hello").ok().expect("failed");
}

fn main() {
    let mut wr = Box::new(io::stdout()) as Box<dyn Write>;
    f(&mut wr);
}
