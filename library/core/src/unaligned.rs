#[cfg_attr(target_arch = "x86_64", repr(packed))]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub(crate) struct Unaligned<T: Copy>(pub T);
