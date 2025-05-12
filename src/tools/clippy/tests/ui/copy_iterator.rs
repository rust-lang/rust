#![warn(clippy::copy_iterator)]
#![allow(clippy::manual_inspect)]

#[derive(Copy, Clone)]
struct Countdown(u8);

impl Iterator for Countdown {
    //~^ copy_iterator

    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        self.0.checked_sub(1).map(|c| {
            self.0 = c;
            c
        })
    }
}

fn main() {
    let my_iterator = Countdown(5);
    assert_eq!(my_iterator.take(1).count(), 1);
    assert_eq!(my_iterator.count(), 5);
}
