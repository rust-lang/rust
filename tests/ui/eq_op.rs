#![feature(plugin)]
#![plugin(clippy)]

#[warn(eq_op)]
#[allow(identity_op, double_parens, many_single_char_names)]
#[allow(no_effect, unused_variables, unnecessary_operation, short_circuit_statement)]
#[warn(nonminimal_bool)]
fn main() {
    // simple values and comparisons
    1 == 1;
    "no" == "no";
    // even though I agree that no means no ;-)
    false != false;
    1.5 < 1.5;
    1u64 >= 1u64;

    // casts, methods, parentheses
    (1 as u64) & (1 as u64);
    1 ^ ((((((1))))));

    // unary and binary operators
    (-(2) < -(2));
    ((1 + 1) & (1 + 1) == (1 + 1) & (1 + 1));
    (1 * 2) + (3 * 4) == 1 * 2 + 3 * 4;

    // various other things
    ([1] != [1]);
    ((1, 2) != (1, 2));
    vec![1, 2, 3] == vec![1, 2, 3]; //no error yet, as we don't match macros

    // const folding
    1 + 1 == 2;
    1 - 1 == 0;

    1 - 1;
    1 / 1;
    true && true;

    true || true;


    let a: u32 = 0;
    let b: u32 = 0;

    a == b && b == a;
    a != b && b != a;
    a < b && b > a;
    a <= b && b >= a;

    let mut a = vec![1];
    a == a;
    2*a.len() == 2*a.len(); // ok, functions
    a.pop() == a.pop(); // ok, functions

    use std::ops::BitAnd;
    struct X(i32);
    impl BitAnd for X {
        type Output = X;
        fn bitand(self, rhs: X) -> X {
            X(self.0 & rhs.0)
        }
    }
    impl<'a> BitAnd<&'a X> for X {
        type Output = X;
        fn bitand(self, rhs: &'a X) -> X {
            X(self.0 & rhs.0)
        }
    }
    let x = X(1);
    let y = X(2);
    let z = x & &y;

    #[derive(Copy, Clone)]
    struct Y(i32);
    impl BitAnd for Y {
        type Output = Y;
        fn bitand(self, rhs: Y) -> Y {
            Y(self.0 & rhs.0)
        }
    }
    impl<'a> BitAnd<&'a Y> for Y {
        type Output = Y;
        fn bitand(self, rhs: &'a Y) -> Y {
            Y(self.0 & rhs.0)
        }
    }
    let x = Y(1);
    let y = Y(2);
    let z = x & &y;
}
