use std::fs::File;
use std::io;
use std::ops::{Deref, DerefMut};

/// A trivial wrapper for [`memmap2::Mmap`] (or `Vec<u8>` on WASM).
#[cfg(not(any(miri, target_arch = "wasm32")))]
pub struct Mmap(memmap2::Mmap);

#[cfg(any(miri, target_arch = "wasm32"))]
pub struct Mmap(Vec<u8>);

#[cfg(not(any(miri, target_arch = "wasm32")))]
impl Mmap {
    /// # Safety
    ///
    /// The given file must not be mutated (i.e., not written, not truncated, ...) until the mapping is closed.
    ///
    /// However in practice most callers do not ensure this, so uses of this function are likely unsound.
    #[inline]
    pub unsafe fn map(file: File) -> io::Result<Self> {
        // By default, memmap2 creates shared mappings, implying that we could see updates to the
        // file through the mapping. That would violate our precondition; so by requesting a
        // map_copy_read_only we do not lose anything.
        // This mapping mode also improves our support for filesystems such as cacheless virtiofs.
        // For more details see https://github.com/rust-lang/rust/issues/122262
        //
        // SAFETY: The caller must ensure that this is safe.
        unsafe { memmap2::MmapOptions::new().map_copy_read_only(&file).map(Mmap) }
    }
}

#[cfg(any(miri, target_arch = "wasm32"))]
impl Mmap {
    #[inline]
    pub unsafe fn map(mut file: File) -> io::Result<Self> {
        use std::io::Read;

        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Mmap(data))
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
