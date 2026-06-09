//@ compile-flags: -Zwrite-long-types-to-disk=yes
// Fixes #110131
//
// The issue is that we were constructing an `ImplDerived` cause code for the
// `&'a T: IntoIterator<Item = &'a u8>` obligation for `Helper::new`, which is
// incorrect because derived obligations are only expected to come from *traits*.

struct SeqBuffer<'a, T>
where
    &'a T: IntoIterator<Item = &'a u8>,
{
    iter: <&'a T as IntoIterator>::IntoIter,
}

struct Helper<'a, T>
where
    &'a T: IntoIterator<Item = &'a u8>,
{
    buf: SeqBuffer<'a, T>,
}

impl<'a, T> Helper<'a, T>
where
    &'a T: IntoIterator<Item = &'a u8>,
{
    fn new(sq: &'a T) -> Self {
        loop {}
    }
}

struct BitReaderWrapper<T>(T);

impl<'a, T> IntoIterator for &'a BitReaderWrapper<T>
where
    &'a T: IntoIterator<Item = &'a u8>,
{
    type Item = u32;

    type IntoIter = Helper<'a, T>;
    //~^ ERROR `Helper<'a, T>` is not an iterator

    fn into_iter(self) -> Self::IntoIter {
        Helper::new(&self.0)
        //~^ ERROR overflow evaluating the requirement `&_: IntoIterator`
    }
}

fn main() {}
