//! This test checks that we do not walk types in async blocks for
//! determining the opaque types that appear in a signature. async blocks,
//! all other coroutines and closures are always private and not part of
//! a signature. They become part of a signature via `dyn Trait` or `impl Trait`,
//! which is something that we process abstractly without looking at its hidden
//! types.
//@ edition: 2021
//@ check-pass

#![feature(impl_trait_in_assoc_type)]

use std::future::Future;

pub struct MemtableLocalStateStore {
    mem_table: MemTable,
}

impl LocalStateStore for MemtableLocalStateStore {
    type IterStream<'a> = impl Sized + 'a where Self: 'a;

    fn iter(&self) -> impl Future<Output = Self::IterStream<'_>> + '_ {
        async move { merge_stream(self.mem_table.iter()) }
    }
}

trait LocalStateStore {
    type IterStream<'a>
    where
        Self: 'a;

    fn iter(&self) -> impl Future<Output = Self::IterStream<'_>> + '_;
}

struct MemTable;

impl MemTable {
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a ()> {
        std::iter::empty()
    }
}

pub(crate) async fn merge_stream<'a>(mem_table_iter: impl Iterator<Item = &'a ()>) {}

fn main() {}
