// Assuming that the hidden type in these tests is `&'_#15r u8`,
// we have a member constraint: `'_#15r member ['static, 'a, 'b, 'c]`.
//
// Make sure we pick up the minimum non-ambiguous region among them.
// We will have to exclude `['b, 'c]` because they're incomparable,
// and then we should pick `'a` because we know `'static: 'a`.

// check-pass

trait Cap<'a> {}
impl<T> Cap<'_> for T {}

fn type_test<'a, T: 'a>() -> &'a u8 { &0 }

// Basic test: make sure we don't bail out because 'b and 'c are incomparable.
fn basic<'a, 'b, 'c>() -> impl Cap<'a> + Cap<'b> + Cap<'c>
where
    'a: 'b,
    'a: 'c,
{
    &0
}

// Make sure we don't pick `'static`.
fn test_static<'a, 'b, 'c, T>() -> impl Cap<'a> + Cap<'b> + Cap<'c>
where
    'a: 'b,
    'a: 'c,
    T: 'a,
{
    type_test::<'_, T>() // This will fail if we pick 'static
}

fn main() {}
