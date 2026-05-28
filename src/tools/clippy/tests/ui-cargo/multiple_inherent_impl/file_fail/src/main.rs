#![allow(dead_code)]
#![deny(clippy::multiple_inherent_impl)]

struct S;

impl S {
    fn first() {}
}

mod a {
    use super::S;

    impl S {
        //^ Must trigger
        fn second() {}
    }
}

mod b {
    struct S;

    impl S {
        fn first() {}
    }

    impl S {
        //^ Must trigger

        fn second() {}
    }
}

mod c;

impl c::S {
    //^ Must NOT trigger
    fn second() {}
}

fn main() {}
