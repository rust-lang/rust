trait Foo {
    fn foo<'a>(x: &i32, y: &'a i32) -> &'a i32;
}

impl Foo for () {
    fn foo<'a>(x: &'a i32, y: &'a i32) -> &'a i32 {
    //~^ ERROR `impl` item signature doesn't match `trait` item signature
        if x > y { x } else { y }
    }
}

fn main() {}
