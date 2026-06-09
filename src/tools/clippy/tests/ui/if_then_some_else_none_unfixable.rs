#![warn(clippy::if_then_some_else_none)]
#![allow(clippy::manual_is_multiple_of)]

mod issue15257 {
    use std::pin::Pin;

    #[derive(Default)]
    pub struct Foo {}
    pub trait Bar {}
    impl Bar for Foo {}

    fn pointer_unsized_coercion(i: u32) -> Option<Box<dyn Bar>> {
        if i % 2 == 0 {
            //~^ if_then_some_else_none
            Some(Box::new(Foo::default()))
        } else {
            None
        }
    }

    fn reborrow_as_pin(i: Pin<&mut i32>) {
        use std::ops::Rem;

        fn do_something(i: Option<&i32>) {
            todo!()
        }

        do_something(if i.rem(2) == 0 {
            //~^ if_then_some_else_none
            Some(&i)
        } else {
            None
        });
    }
}
