#![feature(plugin)]

#![plugin(clippy)]
#![deny(clippy)]

pub fn test(foo: Box<Vec<bool>>) { //~ ERROR You seem to be trying to use Box<Vec<T>>
    println!("{:?}", foo.get(0))
}

fn main(){
    test(Box::new(Vec::new()));
}