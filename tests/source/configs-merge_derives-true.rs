// rustfmt-merge_derives: true
// Merge multiple derives to a single one.

#[bar]
#[derive(Eq, PartialEq)]
#[foo]
#[derive(Debug)]
#[foobar]
#[derive(Copy, Clone)]
pub enum Foo {}
