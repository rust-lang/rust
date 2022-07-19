use std::fs::File;
use std::io;
use std::ops::{Deref, DerefMut};

use crate::owning_ref::StableAddress;

/// A trivial wrapper for [`memmap2::Mmap`] that implements [`StableAddress`].
#[cfg(not(target_arch = "wasm32"))]
pub struct Mmap(memmap2::Mmap);

#[cfg(target_arch = "wasm32")]
pub struct Mmap(Vec<u8>);

#[cfg(not(target_arch = "wasm32"))]
impl Mmap {
    #[inline]
    pub unsafe fn map(file: File) -> io::Result<Self> {
        memmap2::Mmap::map(&file).map(Mmap)
    }
}

#[cfg(target_arch = "wasm32")]
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
        &*self.0
    }
}

// SAFETY: On architectures other than WASM, mmap is used as backing storage. The address of this
// memory map is stable. On WASM, `Vec<u8>` is used as backing storage. The `Mmap` type doesn't
// export any function that can cause the `Vec` to be re-allocated. As such the address of the
// bytes inside this `Vec` is stable.
unsafe impl StableAddress for Mmap {}

#[cfg(not(target_arch = "wasm32"))]
pub struct MmapMut(memmap2::MmapMut);

#[cfg(target_arch = "wasm32")]
pub struct MmapMut(Vec<u8>);

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(target_arch = "wasm32")]
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
        &*self.0
    }
}

impl DerefMut for MmapMut {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut *self.0
    }
}
