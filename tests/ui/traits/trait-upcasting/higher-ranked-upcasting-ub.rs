//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// We previously wrongly instantiated binders during trait upcasting,
// allowing the super trait to be more generic than the sub trait.
// This was unsound.

trait Supertrait<'a, 'b> {
    fn cast(&self, x: &'a str) -> &'b str;
}

trait Subtrait<'a, 'b>: Supertrait<'a, 'b> {}

impl<'a> Supertrait<'a, 'a> for () {
    fn cast(&self, x: &'a str) -> &'a str {
        x
    }
}
impl<'a> Subtrait<'a, 'a> for () {}
fn unsound(x: &dyn for<'a> Subtrait<'a, 'a>) -> &dyn for<'a, 'b> Supertrait<'a, 'b> {
    x //~ ERROR mismatched types
    //[current]~^ ERROR mismatched types
}

fn transmute<'a, 'b>(x: &'a str) -> &'b str {
    unsound(&()).cast(x)
}

fn main() {
    let x;
    {
        let mut temp = String::from("hello there");
        x = transmute(temp.as_str());
    }
    println!("{x}");
}
