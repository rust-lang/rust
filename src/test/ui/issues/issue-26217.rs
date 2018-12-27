fn foo<T>() where for<'a> T: 'a {}

fn main<'a>() {
    foo::<&'a i32>();
    //~^ ERROR the type `&'a i32` does not fulfill the required lifetime
}
