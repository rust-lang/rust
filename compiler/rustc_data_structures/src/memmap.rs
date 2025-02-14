use std::fs::File;
use std::io;
use std::ops::{Deref, DerefMut};
use std::path::Path;

/// A trivial wrapper for [`memmap2::Mmap`] (or `Vec<u8>` on WASM).
#[cfg(not(any(miri, target_arch = "wasm32")))]
pub struct Mmap(memmap2::Mmap);

#[cfg(any(miri, target_arch = "wasm32"))]
pub struct Mmap(Vec<u8>);

#[cfg(not(any(miri, target_arch = "wasm32")))]
impl Mmap {
    /// The given file must not be mutated (i.e., not written, not truncated, ...) until the mapping is closed.
    ///
    /// This process must not modify nor remove the backing file while the memory map lives.
    /// For the dep-graph and the work product index, it is as soon as the decoding is done.
    /// For the query result cache, the memory map is dropped in save_dep_graph before calling
    /// save_in and trying to remove the backing file.
    ///
    /// There is no way to prevent another process from modifying this file.
    ///
    /// This means in practice all uses of this function are theoretically unsound, but also
    /// the way rustc uses `Mmap` (reading bytes, validating them afterwards *anyway* to detect
    /// corrupted files) avoids the actual issues this could cause.
    ///
    /// Someone may truncate our file, but then we'll SIGBUS, which is not great, but at least
    /// we won't succeed with corrupted data.
    ///
    /// To get a bit more hardening out of this we will set the file as readonly before opening it.
    #[inline]
    pub fn map(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let mut perms = std::fs::metadata(path)?.permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(path, perms)?;

        let file = File::open(path)?;

        // By default, memmap2 creates shared mappings, implying that we could see updates to the
        // file through the mapping. That would violate our precondition; so by requesting a
        // map_copy_read_only we do not lose anything.
        // This mapping mode also improves our support for filesystems such as cacheless virtiofs.
        // For more details see https://github.com/rust-lang/rust/issues/122262
        //
        // SAFETY: The caller must ensure that this is safe.
        unsafe { Ok(Self(memmap2::MmapOptions::new().map_copy_read_only(&file)?)) }
    }
}

#[cfg(any(miri, target_arch = "wasm32"))]
impl Mmap {
    #[inline]
    pub fn map(path: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Mmap(std::fs::read(path)?))
    }
}

impl Deref for Mmap {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl AsRef<[u8]> for Mmap {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[cfg(not(any(miri, target_arch = "wasm32")))]
pub struct MmapMut(memmap2::MmapMut);

#[cfg(any(miri, target_arch = "wasm32"))]
pub struct MmapMut(Vec<u8>);

#[cfg(not(any(miri, target_arch = "wasm32")))]
impl MmapMut {
    #[inline]
    pub fn map_anon(len: usize) -> io::Result<Self> {
        let mmap = memmap2::MmapMut::map_anon(len)?;
        Ok(MmapMut(mmap))
    }

    #[inline]
    pub fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }

    #[inline]
    pub fn make_read_only(self) -> std::io::Result<Mmap> {
        let mmap = self.0.make_read_only()?;
        Ok(Mmap(mmap))
    }
}

#[cfg(any(miri, target_arch = "wasm32"))]
impl MmapMut {
    #[inline]
    pub fn map_anon(len: usize) -> io::Result<Self> {
        let data = Vec::with_capacity(len);
        Ok(MmapMut(data))
    }

    #[inline]
    pub fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    #[inline]
    pub fn make_read_only(self) -> std::io::Result<Mmap> {
        Ok(Mmap(self.0))
    }
}

impl Deref for MmapMut {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl DerefMut for MmapMut {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}
