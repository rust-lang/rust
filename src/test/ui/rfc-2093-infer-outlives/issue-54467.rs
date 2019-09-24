// Regression test for #54467:
//
// Here, the trait object has an "inferred outlives" requirement that
// `<Self as MyIterator<'a>>::Item: 'a`; but since we don't know what
// `Self` is, we were (incorrectly) messing things up, leading to
// strange errors. This test ensures that we do not give compilation
// errors.
//
// build-pass (FIXME(62277): could be check-pass?)

trait MyIterator<'a>: Iterator where Self::Item: 'a { }

struct MyStruct<'a, A> {
    item: Box<dyn MyIterator<'a, Item = A>>
}

fn main() { }
