use std;
import std.Option;
import std.Option.t;
import std.Option.none;
import std.Option.some;

fn foo[T](&Option.t[T] y) {
  let int x;
  
  let vec[int] res = vec();
  
  /* tests that x doesn't get put in the precondition for the 
     entire if expression */
  if (true) {
  }
  else {
    alt (y) {
      case (none[T]) {
        x = 17;
      }
      case (_) {
        x = 42;
      }
    }
    res += vec(x);
  }

  ret;
}

fn main() {
  log("hello");
  foo[int](some[int](5));
}