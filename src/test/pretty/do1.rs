// pp-exact

fn f<F>(f: F) where F: Fn(isize) { f(10) }

fn main() { f(|i| { assert_eq!(i , 10) }) }
