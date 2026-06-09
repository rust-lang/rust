// Reproducer from https://github.com/rust-lang/rust/issues/148328#issuecomment-3473688412
#[inline(always)]
pub const fn the_path() -> &'static str {
    std::panic::Location::caller().file()
}

#[inline(never)]
pub fn the_path2() -> &'static str {
    const { std::panic::Location::caller().file() }
}

// Reproducer from https://github.com/rust-lang/rust/issues/148328#issuecomment-3473761194
pub const fn the_path_len() -> usize {
    std::panic::Location::caller().file().len()
}

pub type Array = [u8; the_path_len()];

pub fn the_zeroed_path_len_array() -> Array {
    [0; _]
}
