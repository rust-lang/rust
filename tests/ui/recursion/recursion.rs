//@ build-fail
//@ compile-flags:-C overflow-checks=off --diagnostic-width=100 -Zwrite-long-types-to-disk=yes

enum Nil {NilValue}
struct Cons<T> {head:isize, tail:T}
trait Dot {fn dot(&self, other:Self) -> isize;}
impl Dot for Nil {
  fn dot(&self, _:Nil) -> isize {0}
}
impl<T:Dot> Dot for Cons<T> {
  fn dot(&self, other:Cons<T>) -> isize {
    self.head * other.head + self.tail.dot(other.tail)
  }
}
fn test<T:Dot> (n:isize, i:isize, first:T, second:T) ->isize {
  match n {    0 => {first.dot(second)}
    _ => {test (n-1, i+1, Cons {head:2*i+1, tail:first}, Cons{head:i*i, tail:second})}
    //~^ ERROR recursion limit
  }
}
pub fn main() {
  let n = test(1, 0, Nil::NilValue, Nil::NilValue);
  println!("{}", n);
}
