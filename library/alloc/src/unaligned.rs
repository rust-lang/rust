#[cfg_attr(target_arch = "x86_64", repr(packed))]
pub(crate) struct Unaligned<T: Copy>(pub T);
