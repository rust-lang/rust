//@ run-rustfix
//@ edition:2021
#![deny(unused_qualifications)]
#![deny(unused_imports)]
#![feature(coroutines, coroutine_trait)]

use std::ops::{
    Coroutine,
    CoroutineState::{self, *},
    //~^ ERROR unused import: `*`
};
use std::pin::Pin;

#[allow(dead_code)]
fn finish<T>(mut amt: usize, mut t: T) -> T::Return
    where T: Coroutine<(), Yield = ()> + Unpin,
{
    loop {
        match Pin::new(&mut t).resume(()) {
            CoroutineState::Yielded(()) => amt = amt.checked_sub(1).unwrap(),
            CoroutineState::Complete(ret) => {
                assert_eq!(amt, 0);
                return ret
            }
        }
    }
}


mod foo {
    pub fn bar() {}
}

pub fn main() {

    use foo::bar;
    foo::bar();
    //~^ ERROR unnecessary qualification
    bar();

    // The item `use std::string::String` is imported redundantly.
    // Suppress `unused_imports` reporting, otherwise the fixed file will report an error
    #[allow(unused_imports)]
    use std::string::String;
    let s = String::new();
    let y = std::string::String::new();
    // unnecessary qualification
    println!("{} {}", s, y);

}
