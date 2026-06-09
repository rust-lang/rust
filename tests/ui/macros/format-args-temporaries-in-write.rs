//@ check-fail

use std::fmt::{self, Display};

struct Mutex;

impl Mutex {
    fn lock(&self) -> MutexGuard {
        MutexGuard(self)
    }
}

struct MutexGuard<'a>(&'a Mutex);

impl<'a> Drop for MutexGuard<'a> {
    fn drop(&mut self) {
        // Empty but this is a necessary part of the repro. Otherwise borrow
        // checker is fine with 'a dangling at the time that MutexGuard goes out
        // of scope.
    }
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
    // FIXME(dtolnay): We actually want both of these to work. I think it's
    // sadly unimplementable today though.

    let _write = {
        let mutex = Mutex;
        write!(Out, "{}", mutex.lock()) /* no semicolon */
        //~^ ERROR `mutex` does not live long enough
    };

    let _writeln = {
        let mutex = Mutex;
        writeln!(Out, "{}", mutex.lock()) /* no semicolon */
        //~^ ERROR `mutex` does not live long enough
    };
}
