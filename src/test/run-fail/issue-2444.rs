// error-pattern:explicit failure

extern mod std;
use std::arc;

enum e<T: Const Send> { e(arc::ARC<T>) }

fn foo() -> e<int> {fail;}

fn main() {
   let f = foo();
}
