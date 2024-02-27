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
    //~^ ERROR in type `&'b &'a ()`, reference has a longer lifetime than the data it references
    None::<&'_ &'_ ()>
}

fn main() {
    let y = boom().extend(&String::from("temporary"));
    println!("{}", y);
}
