trait Shadow<'a> {
    type Bar<'a>;
    //~^ ERROR lifetime name `'a` shadows a lifetime name that is already in scope
}

trait NoShadow<'a> {
    type Bar<'b>; // OK
}

impl<'a> NoShadow<'a> for &'a u32 {
    type Bar<'a> = i32;
    //~^ ERROR lifetime name `'a` shadows a lifetime name that is already in scope
}

trait ShadowT<T> {
    type Bar<T>;
    //~^ ERROR the name `T` is already used
}

trait NoShadowT<T> {
    type Bar<U>; // OK
}

impl<T> NoShadowT<T> for Option<T> {
    type Bar<T> = i32;
    //~^ ERROR the name `T` is already used
}

fn main() {}
