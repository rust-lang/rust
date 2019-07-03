// build-pass (FIXME(62277): could be check-pass?)
// edition:2018

#![feature(async_await)]

// This is a regression test to ensure that simple bindings (where replacement arguments aren't
// created during async fn lowering) that have their DefId used during HIR lowering (such as impl
// trait) are visited during def collection and thus have a DefId.

async fn foo(ws: impl Iterator<Item = ()>) {}

fn main() {}
