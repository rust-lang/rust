// run-fail
// error-pattern:returned Box<Error> from main()
// failure-status: 1
// ignore-emscripten no processes

use std::io::{Error, ErrorKind};

fn main() -> Result<(), Box<Error>> {
    Err(Box::new(Error::new(ErrorKind::Other, "returned Box<Error> from main()")))
}
