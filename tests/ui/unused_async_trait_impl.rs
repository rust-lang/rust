#![warn(clippy::unused_async_trait_impl)]

trait HasAsyncMethod {
    async fn do_something() -> u32;
}

struct Stub;

mod typical {
    struct Inefficient;
    struct Efficient;

    use crate::Stub;

    impl crate::HasAsyncMethod for Inefficient {
        async fn do_something() -> u32 {
            //~^ unused_async_trait_impl
            1
        }
    }

    impl crate::HasAsyncMethod for Efficient {
        fn do_something() -> impl Future<Output = u32> {
            std::future::ready(1)
        }
    }

    impl crate::HasAsyncMethod for crate::Stub {
        async fn do_something() -> u32 {
            todo!() // Do not emit the lint in this case.
        }
    }
}

// Test to check if the identation of the various snippets goes as intended.
mod indented {
    struct Indented;

    impl crate::HasAsyncMethod for Indented {
        async fn do_something() -> u32 {
            //~^ unused_async_trait_impl
            let mut x = 0;
            for y in 0..64 {
                x = (x + 1) * y;
            }

            let fake_fut = async {
                if x == 0 {
                    panic!("Fake example");
                }
            };

            x
        }
    }

    struct Complex<T>(std::marker::PhantomData<T>);

    impl<T> crate::HasAsyncMethod for Complex<T>
    where
        T: Sized,
    {
        async fn do_something() -> u32 {
            //~^ unused_async_trait_impl
            5
        }
    }
}

mod default_unchanged {
    trait HasDefaultAsyncMethod {
        // The lint should not suggest a change for trait fn's as changing that decl
        // implies a less restrictive Future type.
        async fn do_something() -> u32 {
            0
        }
    }

    impl HasDefaultAsyncMethod for crate::Stub {
        // Nothing!
    }
}

mod macros {
    trait HasAsyncMethodVec {
        async fn do_something() -> Vec<u32>;
    }

    struct MacroInType;
    struct MacroInExpr;

    macro_rules! vec_ty {
        ($t:ty) => { Vec<$t> }
    }

    impl HasAsyncMethodVec for MacroInType {
        async fn do_something() -> vec_ty!(u32) {
            //~^ unused_async_trait_impl
            Vec::new()
        }
    }

    impl HasAsyncMethodVec for MacroInExpr {
        async fn do_something() -> Vec<u32> {
            //~^ unused_async_trait_impl
            vec![]
        }
    }
}
