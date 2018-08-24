#![feature(generators, generator_trait)]

use std::ops::Generator;

fn main() {
   let s = String::from("foo");
   let mut gen = move || {
   //~^ ERROR the size for values of type
       yield s[..];
   };
   unsafe { gen.resume(); }
   //~^ ERROR the size for values of type
}
