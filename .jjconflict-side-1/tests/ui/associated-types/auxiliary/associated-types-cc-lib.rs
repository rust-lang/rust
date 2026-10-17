// Helper for test issue-18048, which tests associated types in a
// cross-crate scenario.

#![crate_type="lib"]

pub trait Bar: Sized {
    type T;

    fn get(x: Option<Self>) -> <Self as Bar>::T;
}

impl Bar for isize {
    type T = usize;

    fn get(_: Option<isize>) -> usize { 22 }
}
