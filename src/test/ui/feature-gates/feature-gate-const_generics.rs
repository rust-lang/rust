fn foo<const X: ()>() {} //~ ERROR const generics are unstable

struct Foo<const X: usize>([(); X]); //~ ERROR const generics are unstable

macro_rules! accept_item { ($i:item) => {} }
accept_item! {
    impl<const X: ()> A {} //~ ERROR const generics are unstable
}

fn main() {}
