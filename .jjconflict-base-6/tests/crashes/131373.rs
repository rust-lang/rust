//@ known-bug: #131373

trait LockReference: 'static {
    type Ref<'a>;
}

struct SliceRef<'a, T: ?Sized> {
    _x: &'a T,
}

impl<'a, T: ?Sized, SR: LockReference> IntoIterator for SliceRef<'a, T>
where
    &'a T: IntoIterator<Item = &'a SR>,
{
    type Item = SR::Ref<'a>;
    type IntoIter = std::iter::Map<<&'a T as IntoIterator>::IntoIter,
        for<'c> fn(&'c SR) -> SR::Ref<'c>>;
    fn into_iter(self) -> Self::IntoIter {
        loop {}
    }
}

impl LockReference for () {
    type Ref<'a> = ();
}

fn locked() -> SliceRef<'static, [()]> {
    loop {}
}

fn main() {
    let _ = locked().into_iter();
}
