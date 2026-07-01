//@ edition: 2024
//@ check-pass
//@ run-rustfix

#![warn(unused_braces)]

use std::sync::Mutex;

fn consume(lock: &Mutex<Vec<u8>>, _value: usize) {
    let _guard = lock.lock().unwrap();
}

fn consume_int<T>(_: T) {}

struct Lockable(Mutex<Vec<u8>>);

impl Lockable {
    fn update(&self, _value: usize) {
        let _guard = self.0.lock().unwrap();
    }

    fn run(&self) {
        // These blocks shorten the lifetime of the temporary `MutexGuard`.
        consume(&self.0, { self.0.lock().unwrap().len() });
        self.update({ self.0.lock().unwrap().len() });
        consume_int({ [self.0.lock().unwrap().len()] });
    }
}

fn main() {
    let x = 7;

    consume_int({ 7 });
    //~^ WARN unnecessary braces

    consume_int({ x });
    //~^ WARN unnecessary braces

    consume_int({ x as usize });
    //~^ WARN unnecessary braces

    consume_int({ (x, 7) });
    //~^ WARN unnecessary braces

    consume_int({ [x, 7] });
    //~^ WARN unnecessary braces

    Lockable(Mutex::new(vec![1])).update({ 7 });
    //~^ WARN unnecessary braces

    Lockable(Mutex::new(vec![1])).update({ x as usize });
    //~^ WARN unnecessary braces

    Lockable(Mutex::new(vec![1])).run();
}
