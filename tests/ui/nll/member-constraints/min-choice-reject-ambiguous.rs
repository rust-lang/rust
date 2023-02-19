// ... continued from ./min-choice.rs

// check-fail

trait Cap<'a> {}
impl<T> Cap<'_> for T {}

fn type_test<'a, T: 'a>() -> &'a u8 { &0 }

// Make sure we don't pick `'b`.
fn test_b<'a, 'b, 'c, T>() -> impl Cap<'a> + Cap<'b> + Cap<'c>
where
    'a: 'b,
    'a: 'c,
    T: 'b,
{
    type_test::<'_, T>() // This should pass if we pick 'b.
    //~^ ERROR the parameter type `T` may not live long enough
}

// Make sure we don't pick `'c`.
fn test_c<'a, 'b, 'c, T>() -> impl Cap<'a> + Cap<'b> + Cap<'c>
where
    'a: 'b,
    'a: 'c,
    T: 'c,
{
    type_test::<'_, T>() // This should pass if we pick 'c.
    //~^ ERROR the parameter type `T` may not live long enough
}

// We need to pick min_choice from `['b, 'c]`, but it's ambiguous which one to pick because
// they're incomparable.
fn test_ambiguous<'a, 'b, 'c>(s: &'a u8) -> impl Cap<'b> + Cap<'c>
where
    'a: 'b,
    'a: 'c,
{
    s
    //~^ ERROR captures lifetime that does not appear in bounds
}

fn main() {}
