//! Regression test for #114728.

trait Extend<'a, 'b> {
    fn extend(self, _: &'a str) -> &'b str;
}

impl<'a, 'b> Extend<'a, 'b> for Option<&'b &'a ()> {
    fn extend(self, s: &'a str) -> &'b str {
        s
    }
}

fn boom<'a, 'b>() -> impl Extend<'a, 'b> {
    None::<&'_ &'_ ()> //~ ERROR lifetime may not live long enough
}

fn main() {
    let y = boom().extend(&String::from("temporary"));
    println!("{}", y);
}
