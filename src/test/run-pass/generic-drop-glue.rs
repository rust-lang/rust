

fn f<T: Copy>(t: T) { let t1: T = t; }

fn main() { let x = {x: @10, y: @12}; f(x); }
