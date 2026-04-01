//@ compile-flags: --document-private-items

#![deny(rustdoc::redundant_explicit_links)]

mod webdavfs {
    pub struct A;
    pub struct B;
}

/// [`Vfs`][crate::Vfs]
pub use webdavfs::A;
//~^^ error: redundant explicit link target

/// [`Vfs`]
pub use webdavfs::B;

pub struct Vfs;
