//@ run-pass
#![allow(dead_code)]

pub mod test2 {
    // This used to generate an ICE (make sure that default functions are
    // parented to their trait to find the first private thing as the trait).

    struct B;
    trait A { fn foo(&self) {} }
    impl A for B {}

    mod tests {
        use super::A;
        fn foo() {
            let a = super::B;
            a.foo();
        }
    }
}


pub fn main() {}
