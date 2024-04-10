//@ run-pass

#![warn(unused_parens)]
#![allow(dead_code)]

trait Foo {
    fn bar();
    fn tar();
}

macro_rules! unused_parens {
    ($ty:ident) => {
        impl<$ty: Foo> Foo for ($ty,) {
            fn bar() { <$ty>::bar() }
            fn tar() {}
        }
    };

    ($ty:ident $(, $rest:ident)*) => {
        impl<$ty: Foo, $($rest: Foo),*> Foo for ($ty, $($rest),*) {
            fn bar() {
                <$ty>::bar();
                <($($rest),*)>::bar() //~WARN unnecessary parentheses around type
            }
            fn tar() {
              let (_t) = 1; //~WARN unnecessary parentheses around pattern
                            //~| WARN unnecessary parentheses around pattern
              let (_t1,) = (1,);
              let (_t2, _t3) = (1, 2);
            }
        }

        unused_parens!($($rest),*);
    }
}

unused_parens!(T1, T2, T3);

fn main() {}
