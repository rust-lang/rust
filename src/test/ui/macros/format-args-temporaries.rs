// check-pass

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

impl<'a> MutexGuard<'a> {
    fn write_fmt(&self, _args: fmt::Arguments) {}
}

impl<'a> Display for MutexGuard<'a> {
    fn fmt(&self, _formatter: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

fn main() {
    let _write = {
        let out = Mutex;
        let mutex = Mutex;
        write!(out.lock(), "{}", mutex.lock()) /* no semicolon */
    };

    let _writeln = {
        let out = Mutex;
        let mutex = Mutex;
        writeln!(out.lock(), "{}", mutex.lock()) /* no semicolon */
    };

    let _print = {
        let mutex = Mutex;
        print!("{}", mutex.lock()) /* no semicolon */
    };

    let _println = {
        let mutex = Mutex;
        println!("{}", mutex.lock()) /* no semicolon */
    };

    let _eprint = {
        let mutex = Mutex;
        eprint!("{}", mutex.lock()) /* no semicolon */
    };

    let _eprintln = {
        let mutex = Mutex;
        eprintln!("{}", mutex.lock()) /* no semicolon */
    };

    let _panic = {
        let mutex = Mutex;
        panic!("{}", mutex.lock()) /* no semicolon */
    };
}
