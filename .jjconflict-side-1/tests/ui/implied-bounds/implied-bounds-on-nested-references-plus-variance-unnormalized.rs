// Regression test for #129021.

trait ToArg<T> {
    type Arg;
}
impl<T, U> ToArg<T> for U {
    type Arg = T;
}

fn extend_inner<'a, 'b>(x: &'a str) -> <&'b &'a () as ToArg<&'b str>>::Arg { x }
fn extend<'a, 'b>(x: &'a str) -> &'b str {
    (extend_inner as fn(_) -> _)(x)
    //~^ ERROR lifetime may not live long enough
}

fn main() {
    let y = extend(&String::from("Hello World"));
    println!("{}", y);
}
