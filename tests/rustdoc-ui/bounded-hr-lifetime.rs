// This test ensures that rustdoc doesn't panic on higher-ranked lifetimes
// with bounds, because an error should have already been emitted by rustc.

pub fn hrlt<'b, 'c>()
where
    for<'a: 'b + 'c> &'a (): std::fmt::Debug,
    //~^ ERROR lifetime bounds cannot be used in this context
{
}
