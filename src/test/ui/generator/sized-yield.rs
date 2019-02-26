#![feature(generators, generator_trait)]

use std::ops::Generator;
use std::pin::Pin;

fn main() {
   let s = String::from("foo");
   let mut gen = move || {
   //~^ ERROR the size for values of type
       yield s[..];
   };
   Pin::new(&mut gen).resume();
   //~^ ERROR the size for values of type
}
