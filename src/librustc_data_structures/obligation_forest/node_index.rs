use std::num::NonZeroU32;
use std::u32;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NodeIndex {
    index: NonZeroU32,
}

impl NodeIndex {
    #[inline]
    pub fn new(value: usize) -> NodeIndex {
        assert!(value < (u32::MAX as usize));
        NodeIndex { index: NonZeroU32::new((value as u32) + 1).unwrap() }
    }

    #[inline]
    pub fn get(self) -> usize {
        (self.index.get() - 1) as usize
    }
}
