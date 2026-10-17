#![allow(unused_mut)]
#![feature(coroutines, coroutine_trait)]

use std::marker::Unpin;
use std::ops::Coroutine;
use std::ops::CoroutineState::Yielded;
use std::pin::Pin;

pub struct GenIter<G>(G);

impl <G> Iterator for GenIter<G>
where
    G: Coroutine + Unpin,
{
    type Item = G::Yield;

    fn next(&mut self) -> Option<Self::Item> {
        match Pin::new(&mut self.0).resume(()) {
            Yielded(y) => Some(y),
            _ => None
        }
    }
}

fn bug<'a>() -> impl Iterator<Item = &'a str> {
    GenIter(#[coroutine] move || {
        let mut s = String::new();
        yield &s[..] //~ ERROR cannot yield value referencing local variable `s` [E0515]
        //~| ERROR borrow may still be in use when coroutine yields
    })
}

fn main() {
    bug();
}
