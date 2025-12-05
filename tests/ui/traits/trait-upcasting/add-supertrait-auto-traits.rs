//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Target {}
trait Source: Send + Target {}

fn upcast(x: &dyn Source) -> &(dyn Target + Send) { x }

fn same(x: &dyn Source) -> &(dyn Source + Send) { x }

fn main() {}
