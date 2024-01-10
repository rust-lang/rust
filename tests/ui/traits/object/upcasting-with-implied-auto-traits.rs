// check-pass
// revisions: current next
//[next] compile-flags: -Znext-solver

trait Target {}
trait Source: Send + Target {}

fn upcast(x: &dyn Source) -> &(dyn Target + Send) { x }

fn same(x: &dyn Source) -> &(dyn Source + Send) { x }
// ^ This isn't upcasting, just passing dyn through unchanged.

fn main() {}
