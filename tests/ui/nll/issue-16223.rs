// Regression test for #16223: without NLL the `if let` construct together with
// the nested box-structure of `Root` causes an unwanted collateral move.

// The exact error prevented here is:
//
// error[E0382]: use of collaterally moved value: `(root.boxed.rhs as SomeVariant::B).0`
//   --> src/main.rs:55:29
//    |
// 56 |         lhs: SomeVariant::A(a),
//    |                             - value moved here
// 57 |         rhs: SomeVariant::B(b),
//    |                             ^ value used here after move
//    |
//    = note: move occurs because the value has type `A`, which does not implement the `Copy` trait

//@ check-pass

#![feature(box_patterns)]

struct Root {
    boxed: Box<SetOfVariants>,
}

struct SetOfVariants {
    lhs: SomeVariant,
    rhs: SomeVariant,
}

enum SomeVariant {
    A(A),
    B(B),
}

struct A(String);
struct B(String);

fn main() {
    let root = Root {
        boxed: Box::new(SetOfVariants {
            lhs: SomeVariant::A(A(String::from("This is A"))),
            rhs: SomeVariant::B(B(String::from("This is B"))),
        }),
    };
    if let box SetOfVariants {
        lhs: SomeVariant::A(a),
        rhs: SomeVariant::B(b),
    } = root.boxed
    {
        println!("a = {}", a.0);
        println!("b = {}", b.0);
    }
}
