//@ build-pass

pub const fn f<T, const N: usize>(_: [std::mem::MaybeUninit<T>; N]) {}

pub struct Blubb<T>(*const T);

pub const fn g<T, const N: usize>(_: [Blubb<T>; N]) {}

pub struct Blorb<const N: usize>([String; N]);

pub const fn h(_: Blorb<0>) {}

pub struct Wrap(Blorb<0>);

pub const fn i(_: Wrap) {}

fn main() {}
