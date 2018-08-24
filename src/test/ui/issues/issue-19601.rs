#![feature(rustc_attrs)]
#![allow(warnings)]

trait A<T> {}
struct B<T> where B<T>: A<B<T>> { t: T }

#[rustc_error]
fn main() { //~ ERROR compilation successful
}
