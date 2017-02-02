#![feature(plugin)]
#![plugin(clippy)]

#![deny(clippy)]
#![allow(boxed_local)]
#![allow(blacklisted_name)]
#![allow(unused_variables)]
#![allow(dead_code)]

pub fn test1(foo: &Box<bool>) { //~ ERROR you seem to be trying to use `&Box<T>`
    println!("{:?}", foo)
}

pub fn test2() {
    let foo: &Box<bool>; //~ ERROR you seem to be trying to use `&Box<T>`
}

struct Test3<'a> {
    foo: &'a Box<bool> //~ ERROR you seem to be trying to use `&Box<T>`
}

fn main(){
    test1(&Box::new(false));
    test2();
}
