#![warn(clippy::copy_iterator)]

#[derive(Copy, Clone)]
struct Countdown(u8);

impl Iterator for Countdown {
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
    let a: Vec<_> = my_iterator.take(1).collect();
    assert_eq!(a.len(), 1);
    let b: Vec<_> = my_iterator.collect();
    assert_eq!(b.len(), 5);
}
