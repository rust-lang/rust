use std::fmt;

pub trait MethodType {
    type GetProp: ?Sized;
}

pub struct MTFn;

impl<'a> MethodType for MTFn { //~ ERROR E0207
    type GetProp = dyn fmt::Debug + 'a;
}

fn bad(a: Box<<MTFn as MethodType>::GetProp>) -> Box<dyn fmt::Debug+'static> {
    a
}

fn dangling(a: &str) -> Box<dyn fmt::Debug> {
    bad(Box::new(a))
}

fn main() {
    let mut s = "hello".to_string();
    let x = dangling(&s);
    s = String::new();
    println!("{:?}", x);
}
