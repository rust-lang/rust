#![feature(plugin)]

#![plugin(clippy)]
#![deny(clippy)]

macro_rules! boxit {
    ($init:expr, $x:ty) => {
        let _: Box<$x> = Box::new($init);
    }
}

fn test_macro() {
    boxit!(Vec::new(), Vec<u8>);
}
pub fn test(foo: Box<Vec<bool>>) { //~ ERROR you seem to be trying to use `Box<Vec<T>>`
    println!("{:?}", foo.get(0))
}

pub fn test2(foo: Box<Fn(Vec<u32>)>) { // pass if #31 is fixed
    foo(vec![1, 2, 3])
}

fn main(){
    test(Box::new(Vec::new()));
    test2(Box::new(|v| println!("{:?}", v)));
    test_macro();
}
