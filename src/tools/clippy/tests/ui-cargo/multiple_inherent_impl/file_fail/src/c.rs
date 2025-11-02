pub struct S;
struct T;

impl S {
    fn first() {}
}

mod d {
    use super::T;
    impl T {
        fn first() {}
    }
}

mod e {
    use super::T;
    impl T {
        //^ Must trigger
        fn second() {}
    }
}
