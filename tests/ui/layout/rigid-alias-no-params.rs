//@ compile-flags: -O -Cdebug-assertions=on
//@ build-pass

// A regression test for #151791. Computing the layout of
// `<AsOwned as ArchiveWith<'a>>::Archived` fails as the alias
// is still rigid as the where-bound in scope shadows the impl.
//
// This previously caused an incorrect error during MIR optimizations.

struct ArchivedString;

pub trait ArchiveWith<'a> {
    type Archived;
}

struct AsOwned;
impl ArchiveWith<'_> for AsOwned {
    type Archived = ArchivedString;
}

fn foo<'a>()
where
    AsOwned: ArchiveWith<'a>,
{
    let _ = unsafe { &*std::ptr::dangling::<<AsOwned as ArchiveWith<'a>>::Archived>() };
}

fn main() {
    foo();
}
