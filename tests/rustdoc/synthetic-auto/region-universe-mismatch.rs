// issue: <https://github.com/rust-lang/rust/issues/123298>
#![crate_name = "it"]

// Constituent type `<'a> Inner<fn(&'a ())>` doesn't impl `Unpin` since user impl
// candidate `Inner<for<'a> fn(&'a ())>` is less general (the universes don't match).
//
//@ has it/struct.Outer.html \
//      '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//      "impl<'a> !Unpin for Outer<'a>"
pub struct Outer<'a>(Inner<fn(&'a ())>);

struct Inner<T>(T);
impl Unpin for Inner<for<'a> fn(&'a ())> {}

// Trivial counterexample: Here, everything matches perfectly.
//@ has it/struct.Wrap.html \
//      '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//      "impl Unpin for Wrap"
pub struct Wrap(Inner<for<'a> fn(&'a ())>);
