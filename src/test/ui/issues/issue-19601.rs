// compile-pass
// skip-codegen
#![allow(warnings)]
trait A<T> {}
struct B<T> where B<T>: A<B<T>> { t: T }


fn main() {
}
