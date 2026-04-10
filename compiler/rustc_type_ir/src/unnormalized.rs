/// A wrapper for value that needs normalization.
/// FIXME: This is very WIP. The plan is to replace the `skip_normalization` spread
/// throughout the codebase with proper normalization. This is the first step toward
/// switching to eager normalization with the next solver.
#[derive(Clone, PartialOrd, PartialEq, Debug)]
pub struct Unnormalized<T> {
    value: T,
}

impl<T> Unnormalized<T> {
    pub fn new(value: T) -> Unnormalized<T> {
        Unnormalized { value }
    }

    pub fn skip_normalization(self) -> T {
        self.value
    }
}
