// Creating a stack closure which references a box and then
// transferring ownership of the box before invoking the stack
// closure results in a crash.

#![feature(box_syntax)]

fn twice(x: Box<usize>) -> usize {
     *x * 2
}

fn invoke<F>(f: F) where F: FnOnce() -> usize {
     f();
}

fn main() {
      let x  : Box<usize>  = box 9;
      let sq =  || { *x * *x };

      twice(x); //~ ERROR: cannot move out of
      invoke(sq);
}
