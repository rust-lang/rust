// Regression test for issue #84366
// Ensures that proving `closure: 'a` requires that
// we prove `closure_ret_type: 'a`

use std::fmt;
trait Trait {
    type Associated;
}

impl<R, F: Fn() -> R> Trait for F {
    type Associated = R;
}

fn static_transfers_to_associated<T: Trait + 'static>(
    _: &T,
    x: T::Associated,
) -> Box<dyn fmt::Display /* + 'static */>
where
    T::Associated: fmt::Display,
{
    Box::new(x) // T::Associated: 'static follows from T: 'static
}

fn make_static_displayable<'a>(s: &'a str) -> Box<dyn fmt::Display> {
    let f = || -> &'a str { "" };
    // problem is: the closure type of `f` is 'static
    static_transfers_to_associated(&f, s) //~ ERROR borrowed data escapes
}

fn main() {
    let d;
    {
        let x = "Hello World".to_string();
        d = make_static_displayable(&x);
    }
    println!("{}", d);
}
