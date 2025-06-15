// Make sure the generated suggestion suggest editing the user
// code instead of the std macro implementation

//@ run-rustfix

#![allow(dead_code)]

use std::fmt::{self, Display};

struct Mutex;

impl Mutex {
    fn lock(&self) -> MutexGuard<'_> {
        MutexGuard(self)
    }
}

struct MutexGuard<'a>(&'a Mutex);

impl<'a> Drop for MutexGuard<'a> {
    fn drop(&mut self) {}
}

struct Out;

impl Out {
    fn write_fmt(&self, _args: fmt::Arguments) {}
}

impl<'a> Display for MutexGuard<'a> {
    fn fmt(&self, _formatter: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

fn main() {
    let _write = {
        let mutex = Mutex;
        write!(Out, "{}", mutex.lock())
        //~^ ERROR `mutex` does not live long enough
        //~| SUGGESTION ;
    };

    let _write = {
        use std::io::Write as _;

        let mutex = Mutex;
        write!(std::io::stdout(), "{}", mutex.lock())
        //~^ ERROR `mutex` does not live long enough
        //~| SUGGESTION let x
    };
}
