// issue: rust-lang/rust#124022

struct Type<T>;
//~^ ERROR type parameter `T` is never used

fn main() {
    {
        impl<T> Type<T> {
            fn new() -> Type<T> {
                Type
                //~^ ERROR type annotations needed
            }
        }
    };
}
