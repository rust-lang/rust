//@ check-pass
//@ compile-flags: --crate-type=lib
//@ revisions: old next
//@[next] compile-flags: -Znext-solver

use std::marker::PhantomData;

pub trait ZeroMapKV<'a> {
    type Container;
}

pub trait ZeroFrom<'zf, C: ?Sized> {}

pub struct ZeroMap<'a, K: ZeroMapKV<'a>>(PhantomData<&'a K>);

impl<'zf, 's, K> ZeroFrom<'zf, ZeroMap<'s, K>> for ZeroMap<'zf, K>
where
    K: for<'b> ZeroMapKV<'b>,
    <K as ZeroMapKV<'zf>>::Container: ZeroFrom<'zf, <K as ZeroMapKV<'s>>::Container>,
{
}
