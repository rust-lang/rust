fn impure(_i: int) {}

// check that unchecked alone does not override borrowck:
fn foo(v: &const Option<int>) {
    match *v {
      Some(ref i) => {
        //~^ ERROR illegal borrow unless pure
        unsafe {
            impure(*i); //~ NOTE impure due to access to impure function
        }
      }
      None => {
      }
    }
}

fn bar(v: &const Option<int>) {
    match *v {
      Some(ref i) => {
        unsafe {
            impure(*i);
        }
      }
      None => {
      }
    }
}

fn main() {
}
