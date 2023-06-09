// Nested impl-traits can impose different member constraints on the same region variable.

// check-pass

trait Cap<'a> {}
impl<T> Cap<'_> for T {}

// Assuming the hidden type is `[&'?15 u8; 1]`, we have two distinct member constraints:
// - '?15 member ['static, 'a, 'b] // from outer impl-trait
// - '?15 member ['static, 'a]     // from inner impl-trait
// To satisfy both we can only choose 'a.
fn pass_early_bound<'s, 'a, 'b>(a: &'s u8) -> impl IntoIterator<Item = impl Cap<'a>> + Cap<'b>
where
    's: 'a,
    's: 'b,
{
    [a]
}

// Same as the above but with late-bound regions.
fn pass_late_bound<'s, 'a, 'b>(
    a: &'s u8,
    _: &'a &'s u8,
    _: &'b &'s u8,
) -> impl IntoIterator<Item = impl Cap<'a>> + Cap<'b> {
    [a]
}

fn main() {}
