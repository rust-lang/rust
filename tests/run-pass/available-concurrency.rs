#![feature(available_concurrency)]

fn main() {
    assert_eq!(std::thread::available_concurrency().unwrap().get(), 1);
}
