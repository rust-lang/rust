// Regression test for #112201. This recursive call previously meant that
// we delay an error when checking opaques at the end of writeback but don't
// encounter that incorrect defining use during borrowck as it's in dead code.

pub fn wrap<T>(x: T) -> impl Sized {
    x
}

fn repeat_helper<T>(x: T) -> impl Sized {
    return x;
    repeat_helper(wrap(x))
    //~^ ERROR expected generic type parameter, found `impl Sized`
    //~| ERROR type parameter `T` is part of concrete type
}


fn main() {}
