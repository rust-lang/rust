//@ known-bug: #140365
//@compile-flags: -C opt-level=1 -Zvalidate-mir
fn f() -> &'static str
where
    Self: Sized,
{
    ""
}
