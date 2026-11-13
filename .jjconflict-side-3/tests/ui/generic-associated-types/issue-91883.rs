//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

use std::fmt::Debug;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct TransactionImpl<'db> {
    _marker: PhantomData<&'db ()>,
}

#[derive(Debug)]
pub struct CursorImpl<'txn> {
    _marker: PhantomData<&'txn ()>,
}

pub trait Cursor<'txn> {}

pub trait Transaction<'db>: Send + Sync + Debug + Sized {
    type Cursor<'tx>: Cursor<'tx>
    where
        'db: 'tx,
        Self: 'tx;

    fn cursor<'tx>(&'tx self) -> Result<Self::Cursor<'tx>, ()>
    where
        'db: 'tx;
}

impl<'tx> Cursor<'tx> for CursorImpl<'tx> {}

impl<'db> Transaction<'db> for TransactionImpl<'db> {
    type Cursor<'tx> = CursorImpl<'tx>; //~ ERROR lifetime bound not satisfied

    fn cursor<'tx>(&'tx self) -> Result<Self::Cursor<'tx>, ()>
    where
        'db: 'tx,
    {
        loop {}
    }
}

fn main() {}
