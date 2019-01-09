// pretty-expanded FIXME #23616

#![feature(box_syntax)]

fn leaky<T>(_t: T) { }

pub fn main() { let x = box 10; leaky::<Box<isize>>(x); }
