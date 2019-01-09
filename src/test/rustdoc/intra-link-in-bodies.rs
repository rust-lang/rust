// we need to make sure that intra-doc links on trait impls get resolved in the right scope

#![deny(intra_doc_link_resolution_failure)]

pub mod inner {
    pub struct SomethingOutOfScope;
}

pub mod other {
    use inner::SomethingOutOfScope;
    use SomeTrait;

    pub struct OtherStruct;

    /// Let's link to [SomethingOutOfScope] while we're at it.
    impl SomeTrait for OtherStruct {}
}

pub trait SomeTrait {}

pub struct SomeStruct;

fn __implementation_details() {
    use inner::SomethingOutOfScope;

    // FIXME: intra-links resolve in their nearest module scope, not their actual scope in cases
    // like this
    // Let's link to [SomethingOutOfScope] while we're at it.
    impl SomeTrait for SomeStruct {}
}
