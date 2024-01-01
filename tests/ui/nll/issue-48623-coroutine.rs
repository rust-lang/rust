// build-pass

#![allow(path_statements)]
#![feature(coroutines, coroutine_trait)]

struct WithDrop;

impl Drop for WithDrop {
    fn drop(&mut self) {}
}

fn reborrow_from_coroutine(r: &mut ()) {
    let d = WithDrop;
    move || { d; yield; &mut *r };
}

fn main() {}
