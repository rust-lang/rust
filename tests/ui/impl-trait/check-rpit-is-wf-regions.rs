type Static<'a> = &'static &'a ();

trait Extend<'a> {
    fn extend(self, _: &'a str) -> &'static str;
}

impl<'a> Extend<'a> for Static<'a> {
    fn extend(self, s: &'a str) -> &'static str {
        s
    }
}

fn boom<'a>(arg: Static<'_>) -> impl Extend<'a> {
    //~^ ERROR in type `&'static &'a ()`, reference has a longer lifetime than the data it references
    arg
}

fn main() {
    let y = boom(&&()).extend(&String::from("temporary"));
    println!("{}", y);
}
