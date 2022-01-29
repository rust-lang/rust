fn foo<T>() where for<'a> T: 'a {}

fn bar<'a>() {
    foo::<&'a i32>();
    //~^ ERROR the type `&'a i32` does not fulfill the required lifetime
}

fn main() {
    bar();
}
