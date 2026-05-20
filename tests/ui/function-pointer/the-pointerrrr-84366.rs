use std::fmt;

trait Trait: 'static {
    type Associated;
}

impl<R, F: (Fn() -> R) + 'static> Trait for F {
    type Associated = R;
}

// NOTE: this is early-bound, see https://rustc-dev-guide.rust-lang.org/early-late-parameters.html#must-be-constrained-by-argument-types
fn early_bound_function<'evil_intimidating>() -> &'evil_intimidating str { "" }

fn static_transfers_to_associated<T: Trait + 'static>(
    _: &T,
    // it is assumed that the associated value is also static
    x: T::Associated,
) -> Box<dyn fmt::Display + 'static>
where
    T::Associated: fmt::Display,
{
    Box::new(x)
}

fn require_static<F: 'static>(_: F) {}

fn make_static_displayable<'temp>(not_static: &'temp str) -> Box<dyn fmt::Display> {
    require_static(early_bound_function);
    static_transfers_to_associated(&early_bound_function, not_static)
    //~^ ERROR borrowed data escapes outside of function [E0521]
}

// FIXME: add a test for closures, those are currently broken
// fn make_static_displayable_closure<'temp>(not_static: &'temp str) -> Box<dyn fmt::Display> {
//     let closure = || -> &'temp str { "" };
//     require_static(closure);
//     static_transfers_to_associated(&closure, not_static)
//     // ERROR borrowed data escapes outside of function [E0521]
// }

fn main() {
    let d;
    {
        let x = "Hello World".to_string();
        d = make_static_displayable(&x);
    }
    println!("{}", d);
}
