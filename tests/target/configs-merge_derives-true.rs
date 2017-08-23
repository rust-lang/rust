// rustfmt-merge_derives: true
// Merge multiple derives to a single one.

#[bar]
#[foo]
#[foobar]
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum Foo {}
