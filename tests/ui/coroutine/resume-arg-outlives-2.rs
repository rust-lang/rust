// Regression test for 132104

#![feature(coroutine_trait, coroutines)]

use std::ops::Coroutine;
use std::{thread, time};

fn demo<'not_static>(s: &'not_static str) -> thread::JoinHandle<()> {
    let mut generator = Box::pin({
        #[coroutine]
        move |_ctx| {
            let ctx: &'not_static str = yield;
            yield;
            dbg!(ctx);
        }
    });

    // exploit:
    generator.as_mut().resume("");
    generator.as_mut().resume(s); // <- generator hoards it as `let ctx`.
    thread::spawn(move || {
        //~^ ERROR borrowed data escapes outside of function
        thread::sleep(time::Duration::from_millis(200));
        generator.as_mut().resume(""); // <- resumes from the last `yield`, running `dbg!(ctx)`.
    })
}

fn main() {
    let local = String::from("...");
    let thread = demo(&local);
    drop(local);
    let _unrelated = String::from("UAF");
    thread.join().unwrap();
}
