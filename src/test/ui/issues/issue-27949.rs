// run-pass
//
// At one time, the `==` operator (and other binary operators) did not
// support subtyping during type checking, and would therefore require
// LHS and RHS to be exactly identical--i.e. to have the same lifetimes.
//
// This was fixed in 1a7fb7dc78439a704f024609ce3dc0beb1386552.

#[derive(Copy, Clone)]
struct Input<'a> {
    foo: &'a u32
}

impl <'a> std::cmp::PartialEq<Input<'a>> for Input<'a> {
    fn eq(&self, other: &Input<'a>) -> bool {
        self.foo == other.foo
    }

    fn ne(&self, other: &Input<'a>) -> bool {
        self.foo != other.foo
    }
}


fn check_equal<'a, 'b>(x: Input<'a>, y: Input<'b>) -> bool {
    // Type checking error due to 'a != 'b prior to 1a7fb7dc78
    x == y
}

fn main() {
    let i = 1u32;
    let j = 1u32;
    let k = 2u32;

    let input_i = Input { foo: &i };
    let input_j = Input { foo: &j };
    let input_k = Input { foo: &k };
    assert!(check_equal(input_i, input_i));
    assert!(check_equal(input_i, input_j));
    assert!(!check_equal(input_i, input_k));
}
