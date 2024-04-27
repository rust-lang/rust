//@ check-pass
//@ known-bug: #100051

// Should fail. Implied bounds from projections in impl headers can create
// improper lifetimes.  Variant of issue #98543 which was fixed by #99217.

trait Trait {
    type Type;
}

impl<T> Trait for T {
    type Type = ();
}

trait Extend<'a, 'b> {
    fn extend(self, s: &'a str) -> &'b str;
}

impl<'a, 'b> Extend<'a, 'b> for <&'b &'a () as Trait>::Type
where
    for<'what, 'ever> &'what &'ever (): Trait,
{
    fn extend(self, s: &'a str) -> &'b str {
        s
    }
}

fn main() {
    let y = <() as Extend<'_, '_>>::extend((), &String::from("Hello World"));
    println!("{}", y);
}
