use core::nonzero::NonZero;
use std::u32;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NodeIndex {
    index: NonZero<u32>
}

impl NodeIndex {
    pub fn new(value: usize) -> NodeIndex {
        assert!(value < (u32::MAX as usize));
        unsafe {
            NodeIndex { index: NonZero::new((value as u32) + 1) }
        }
    }

    pub fn get(self) -> usize {
        (*self.index - 1) as usize
    }
}

