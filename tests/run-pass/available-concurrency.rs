#![feature(available_parallelism)]

fn main() {
    assert_eq!(std::thread::available_parallelism().unwrap().get(), 1);
}
