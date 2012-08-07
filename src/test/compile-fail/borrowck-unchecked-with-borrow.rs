fn impure(_i: int) {}

// check that unchecked alone does not override borrowck:
fn foo(v: &const option<int>) {
    match *v {
      some(ref i) => {
        //~^ ERROR illegal borrow unless pure
        unchecked {
            impure(*i); //~ NOTE impure due to access to impure function
        }
      }
      none => {
      }
    }
}

fn bar(v: &const option<int>) {
    match *v {
      some(ref i) => {
        unsafe {
            impure(*i);
        }
      }
      none => {
      }
    }
}

fn main() {
}
