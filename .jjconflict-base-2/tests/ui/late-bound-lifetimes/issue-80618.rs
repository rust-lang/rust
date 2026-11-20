fn foo<'a>(x: &'a str) -> &'a str {
    x
}

fn main() {
    let _ = foo::<'static>;
//~^ ERROR cannot specify lifetime arguments explicitly if late bound lifetime parameters are present [E0794]
}
