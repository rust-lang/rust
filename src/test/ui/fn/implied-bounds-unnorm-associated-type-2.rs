// check-fail

trait Trait {
    type Type;
}

impl<T> Trait for T {
    type Type = ();
}

fn f<'a, 'b>(_: <&'a &'b () as Trait>::Type)
//~^ ERROR in type `&'a &'b ()`, reference has a longer lifetime than the data it references
where
    'a: 'a,
    'b: 'b,
{
}

fn g<'a, 'b>() {
    f::<'a, 'b>(());
}

fn main() {}
