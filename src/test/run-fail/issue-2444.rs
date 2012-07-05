// error-pattern:explicit failure

enum e<T: const send> { e(arc::arc<T>) }

fn foo() -> e<int> {fail;}

fn main() {
   let f = foo();
}
