pub use imp::fs as imp;

pub mod traits {
    pub use super::{Fs as sys_Fs, DirEntry as sys_DirEntry, OpenOptions as sys_OpenOptions, FilePermissions as sys_FilePermissions, FileType as sys_FileType, DirBuilder as sys_DirBuilder, File as sys_File, FileAttr as sys_FileAttr};
}

pub mod prelude {
    pub use super::imp::Fs;
    pub use super::traits::*;

    pub type ReadDir = <Fs as sys_Fs>::ReadDir;
    pub type File = <Fs as sys_Fs>::File;
    pub type FileAttr = <Fs as sys_Fs>::FileAttr;
    pub type DirEntry = <Fs as sys_Fs>::DirEntry;
    pub type OpenOptions = <Fs as sys_Fs>::OpenOptions;
    pub type FilePermissions = <Fs as sys_Fs>::FilePermissions;
    pub type FileType = <Fs as sys_Fs>::FileType;
    pub type DirBuilder = <Fs as sys_Fs>::DirBuilder;
    pub type FileHandle = <Fs as sys_Fs>::FileHandle;
    pub type Mode = <Fs as sys_Fs>::Mode;
    pub type INode = <Fs as sys_Fs>::INode;
}

use os_str::prelude::*;
use error::prelude::*;
use inner::prelude::*;
use c_str::CStr;
use io;
use core::fmt;
use core::hash;

pub trait ReadDir<F: Fs + ?Sized> : Iterator<Item=Result<F::DirEntry>> { }

pub trait DirEntry<F: Fs + ?Sized> {
    fn file_name(&self) -> &OsStr;
    fn root(&self) -> &OsStr;
    fn metadata(&self) -> Result<F::FileAttr>;
    fn file_type(&self) -> Result<F::FileType>;
    fn ino(&self) -> F::INode;
}

pub trait OpenOptions<F: Fs + ?Sized>: Clone {
    fn new() -> Self where Self: Sized;

    fn read(&mut self, read: bool);
    fn write(&mut self, write: bool);
    fn append(&mut self, append: bool);
    fn truncate(&mut self, truncate: bool);
    fn create(&mut self, create: bool);
    fn mode(&mut self, mode: F::Mode);
}

pub trait FilePermissions<F: Fs + ?Sized>: Sized + Clone + PartialEq + fmt::Debug {
    fn readonly(&self) -> bool;
    fn set_readonly(&mut self, readonly: bool);
    fn mode(&self) -> F::Mode;
}

pub trait FileType<F: Fs + ?Sized>: Copy + Clone + PartialEq + Eq + hash::Hash {
    fn is_dir(&self) -> bool;
    fn is_file(&self) -> bool;
    fn is_symlink(&self) -> bool;

    fn is(&self, mode: F::Mode) -> bool;
}

pub trait DirBuilder<F: Fs + ?Sized> {
    fn new() -> Self where Self: Sized;
    fn mkdir(&self, p: &OsStr) -> Result<()>;

    fn set_mode(&mut self, mode: F::Mode);
}

pub trait File<F: Fs + ?Sized>: io::Read + io::Write + io::Seek + AsInner<F::FileHandle> + IntoInner<F::FileHandle> + fmt::Debug {
    fn open(path: &OsStr, opts: &F::OpenOptions) -> Result<Self> where Self: Sized;
    fn open_c(path: &CStr, opts: &F::OpenOptions) -> Result<Self> where Self: Sized;

    fn fsync(&self) -> Result<()>;
    fn datasync(&self) -> Result<()>;
    fn truncate(&self, sz: u64) -> Result<()>;
    fn file_attr(&self) -> Result<F::FileAttr>;
}

pub trait FileAttr<F: Fs + ?Sized> {
    fn size(&self) -> u64;
    fn perm(&self) -> F::FilePermissions;
    fn file_type(&self) -> F::FileType;
}

pub trait Fs {
    type ReadDir: ReadDir<Self>;
    type File: File<Self>;
    type FileAttr: FileAttr<Self>;
    type DirEntry: DirEntry<Self>;
    type OpenOptions: OpenOptions<Self>;
    type FilePermissions: FilePermissions<Self>;
    type FileType: FileType<Self>;
    type DirBuilder: DirBuilder<Self>;
    type FileHandle;
    type Mode;
    type INode;

    const COPY_IMP: bool;
    fn copy(from: &OsStr, to: &OsStr) -> Result<u64>;

    fn unlink(p: &OsStr) -> Result<()>;
    fn stat(p: &OsStr) -> Result<Self::FileAttr>;
    fn lstat(p: &OsStr) -> Result<Self::FileAttr>;
    fn rename(from: &OsStr, to: &OsStr) -> Result<()>;
    fn link(src: &OsStr, dst: &OsStr) -> Result<()>;
    fn symlink(src: &OsStr, dst: &OsStr) -> Result<()>;
    fn readlink(p: &OsStr) -> Result<OsString>;
    fn canonicalize(p: &OsStr) -> Result<OsString>;
    fn rmdir(p: &OsStr) -> Result<()>;
    fn readdir(p: &OsStr) -> Result<Self::ReadDir>;
    fn set_perm(p: &OsStr, perm: Self::FilePermissions) -> Result<()>;
}
