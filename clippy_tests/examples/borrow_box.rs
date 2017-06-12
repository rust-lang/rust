#![feature(plugin)]
#![plugin(clippy)]

#![deny(borrowed_box)]
#![allow(blacklisted_name)]
#![allow(unused_variables)]
#![allow(dead_code)]

pub fn test1(foo: &mut Box<bool>) {
    println!("{:?}", foo)
}

pub fn test2() {
    let foo: &Box<bool>;
}

struct Test3<'a> {
    foo: &'a Box<bool>
}

trait Test4 {
    fn test4(a: &Box<bool>);
}

impl<'a> Test4 for Test3<'a> {
    fn test4(a: &Box<bool>) {
        unimplemented!();
    }
}

fn main(){
    test1(&mut Box::new(false));
    test2();
}
