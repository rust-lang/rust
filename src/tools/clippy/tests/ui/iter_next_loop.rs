#![allow(dead_code, unused, for_loops_over_fallibles)]
#![warn(clippy::iter_next_loop)]

fn main() {
    let x = [1, 2, 3, 4];
    for _ in vec.iter().next() {}

    struct Unrelated(&'static [u8]);
    impl Unrelated {
        fn next(&self) -> std::slice::Iter<u8> {
            self.0.iter()
        }
    }
    let u = Unrelated(&[0]);
    for _v in u.next() {} // no error
}
