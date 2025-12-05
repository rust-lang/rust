trait Trait<'a>: 'a {
    type Type;
}

// if the `T: 'a` bound gets implied we would probably get ub here again
impl<'a, T> Trait<'a> for T {
    //~^ ERROR the parameter type `T` may not live long enough
    //~| ERROR the parameter type `T` may not live long enough
    type Type = ();
}

fn f<'a, 'b>(s: &'b str, _: <&'b () as Trait<'a>>::Type) -> &'a str
where
    &'b (): Trait<'a>,
{
    s
}

fn main() {
    let x = String::from("Hello World!");
    let y = f(&x, ());
    drop(x); //~ ERROR cannot move out of `x`
    println!("{}", y);
}
