use std::io::{Error, ErrorKind};
use std::thread;

struct T1;
struct T2;

fn main() {
    (Error::new(ErrorKind::Other, "1"), T1, 1) == (Error::new(ErrorKind::Other, "1"), T1, 2);
    //~^ ERROR: can't compare
    (Error::new(ErrorKind::Other, "2"), thread::current())
        == (Error::new(ErrorKind::Other, "2"), thread::current());
    //~^ ERROR: can't compare
    (Error::new(ErrorKind::Other, "4"), thread::current(), T1, T2)
        == (Error::new(ErrorKind::Other, "4"), thread::current(), T1, T2);
    //~^ ERROR: can't compare
}
