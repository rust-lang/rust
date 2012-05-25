// xfail-fast  (compile-flags unsupported on windows)
// compile-flags:--borrowck=err

fn impure(_i: int) {}

// check that unchecked alone does not override borrowck:
fn foo(v: &const option<int>) {
    alt *v {
      some(i) {
        //!^ ERROR illegal borrow unless pure: enum variant in aliasable, mutable location
        unchecked {
            impure(i); //! NOTE impure due to access to impure function
        }
      }
      none {
      }
    }
}

fn bar(v: &const option<int>) {
    alt *v {
      some(i) {
        unsafe {
            impure(i);
        }
      }
      none {
      }
    }
}

fn main() {
}