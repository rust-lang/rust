// edition: 2021

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
