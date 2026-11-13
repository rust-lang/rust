//@check-pass
#![feature(trivial_bounds)]

struct Helper<T>(T);

trait Unsized<T: ?Sized> {
    const SIZE: usize = usize::MAX;
}

impl<T: ?Sized> Unsized<T> for T {}

impl<T> Helper<T> {
    const SIZE: usize = size_of::<T>();
}

struct TrickClippy(str);

impl TrickClippy {
    fn trick_clippy() -> bool
    where
        Self: Sized,
    {
        Helper::<Self>::SIZE == str::SIZE
    }
}

fn main() {}
