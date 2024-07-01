//@ known-bug: unknown
//@ failure-status: 101

struct Foo {}

impl<const UNUSED: usize> Drop for Foo {}

fn main() {}
