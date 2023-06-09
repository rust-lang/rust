// check-pass
// known-bug: #84591

// Should fail. Subtrait can incorrectly extend supertrait lifetimes even when
// supertrait has weaker implied bounds than subtrait. Strongly related to
// issue #25860.

trait Subtrait<T>: Supertrait {}
trait Supertrait {
    fn action(self);
}

fn subs_to_soup<T, U>(x: T)
where
    T: Subtrait<U>,
{
    soup(x)
}

fn soup<T: Supertrait>(x: T) {
    x.action();
}

impl<'a, 'b: 'a> Supertrait for (&'b str, &mut &'a str) {
    fn action(self) {
        *self.1 = self.0;
    }
}

impl<'a, 'b> Subtrait<&'a &'b str> for (&'b str, &mut &'a str) {}

fn main() {
    let mut d = "hi";
    {
        let x = "Hello World".to_string();
        subs_to_soup((x.as_str(), &mut d));
    }
    println!("{}", d);
}
