#![warn(clippy::indexing_slicing)]

/// An opaque integer representation
pub struct Integer<'a> {
    /// The underlying data
    value: &'a [u8],
}
impl<'a> Integer<'a> {
    // Check whether `self` holds a negative number or not
    pub const fn is_negative(&self) -> bool {
        self.value[0] & 0b1000_0000 != 0
    }
}

fn main() {}
