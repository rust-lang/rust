// run-fail
//@error-in-other-file:returned Box<Error> from main()
// failure-status: 1
//@ignore-target-emscripten no processes

use std::io::{Error, ErrorKind};

fn main() -> Result<(), Box<Error>> {
    Err(Box::new(Error::new(ErrorKind::Other, "returned Box<Error> from main()")))
}
