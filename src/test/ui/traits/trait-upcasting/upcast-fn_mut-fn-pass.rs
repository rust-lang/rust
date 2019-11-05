// run-pass

#![feature(trait_upcasting)]

fn call_it<T>(f: Box<dyn FnOnce() -> T>) -> T {
    f()
}

fn main() {
    let f: Box<dyn FnMut() -> i32> = Box::new(|| 42);
    assert_eq!(call_it(f), 42);
}
