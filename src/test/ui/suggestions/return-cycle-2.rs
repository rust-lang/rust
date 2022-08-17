use std::marker::PhantomData;

struct Token<T>(PhantomData<T>);

impl<T> Token<T> {
    fn as_ref(_: i32, _: i32) -> _ {
        //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
        //~| NOTE not allowed in type signatures
        //~| HELP replace with the correct return type
        Token(PhantomData::<&T>)
    }
}

fn main() {}
