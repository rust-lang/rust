// xfail-fast  (compile-flags unsupported on windows)
// compile-flags:--borrowck=err

fn impure(_i: int) {}

fn foo(v: &const option<int>) {
    alt *v {
      some(i) {
        //!^ NOTE pure context is required due to an illegal borrow: enum variant in aliasable, mutable location
        // check that unchecked alone does not override borrowck:
        unchecked {
            impure(i); //! ERROR access to non-pure functions prohibited in a pure context
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