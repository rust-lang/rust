// run-pass
#![feature(trait_upcasting)]
#![allow(incomplete_features)]

struct Test {
    func: Box<dyn FnMut() + 'static>,
}

fn main() {
    let closure: Box<dyn Fn() + 'static> = Box::new(|| ());
    let mut test = Box::new(Test { func: closure });
    (test.func)();
}
