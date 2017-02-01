#![feature(plugin)]
#![plugin(clippy)]

#![deny(clippy)]
#![allow(boxed_local)]
#![allow(blacklisted_name)]

pub fn test(foo: &Box<bool>) { //~ ERROR you seem to be trying to use `&Box<T>`
    println!("{:?}", foo)
}

fn main(){
    test(&Box::new(false));
}
