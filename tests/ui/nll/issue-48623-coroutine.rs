//@ run-pass
#![allow(path_statements)]
#![allow(dead_code)]

#![feature(coroutines, coroutine_trait)]

struct WithDrop;

impl Drop for WithDrop {
    fn drop(&mut self) {}
}

fn reborrow_from_coroutine(r: &mut ()) {
    let d = WithDrop;
    #[coroutine] move || { d; yield; &mut *r }; //~ WARN unused coroutine that must be used
}

fn main() {}
