// Nested impl-traits can impose different member constraints on the same region variable.

// check-fail

trait Cap<'a> {}
impl<T> Cap<'_> for T {}

// Assuming the hidden type is `[&'_#15r u8; 1]`, we have two distinct member constraints:
// - '_#15r member ['static, 'a, 'b] // from outer impl-trait
// - '_#15r member ['static, 'a, 'b] // from inner impl-trait
// To satisfy both we can choose 'a or 'b, so it's a failure due to ambiguity.
fn fail_early_bound<'s, 'a, 'b>(a: &'s u8) -> impl IntoIterator<Item = impl Cap<'a> + Cap<'b>>
where
    's: 'a,
    's: 'b,
{
    [a]
    //~^ E0700
    //~| E0700
}

// Same as the above but with late-bound regions.
fn fail_late_bound<'s, 'a, 'b>(
    a: &'s u8,
    _: &'a &'s u8,
    _: &'b &'s u8,
) -> impl IntoIterator<Item = impl Cap<'a> + Cap<'b>> {
    [a]
    //~^ E0700
    //~| E0700
}

fn main() {}
