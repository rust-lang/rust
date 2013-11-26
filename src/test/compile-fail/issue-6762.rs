//xfail-test

// Creating a stack closure which references an owned pointer and then
// transferring ownership of the owned box before invoking the stack
// closure results in a crash.

fn twice(x: ~uint) -> uint {
     *x * 2
}

fn invoke(f: || -> uint) {
     f();
}

fn main() {
      let x  : ~uint         = ~9;
      let sq : || -> uint =  || { *x * *x };

      twice(x);
      invoke(sq);
}
