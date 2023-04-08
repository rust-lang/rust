use std::borrow::Cow;
use std::alloc::{Allocator, Global, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use rustc_abi::{Size, Align};
use rustc_middle::mir::interpret::AllocBytes;

enum FillBytes<'a> {
    Bytes(&'a [u8]),
    Zero(Size),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AlignedBytes(NonNull<[u8]>, Align);

impl AlignedBytes {
    fn alloc(fill: FillBytes<'_>, align: Align) -> Option<Self> {
        let len = match FillBytes {
            FillBytes::Bytes(b) => b.len(),
            FillBytes::Zero(s) => s.bytes() as usize,
        };

        let layout = Layout::from_size_align(len, align.bytes() as usize)
            .unwrap();
        let mut bytes = Global.allocate_zeroed(layout)
            .ok()?;

        let slice = unsafe { bytes.as_mut() };
        match fill {
            FillBytes::Bytes(data) => slice.copy_from_slice(data),
            FillBytes::Zero(_) => (),
        }

        debug_assert_eq!(bytes.as_ptr() as usize % align.bytes() as usize, 0);

        Some(Self(bytes, align))
    }
}

impl Deref for AlignedBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}

impl DerefMut for AlignedBytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.as_mut() }
    }
}

impl AllocBytes for AlignedBytes {
    fn adjust_to_align(self, align: Align) -> Self {
        if self.align >= align {
            self
        } else {
            let out = Self::alloc(FillBytes::Bytes(&*self), align)
                .unwrap();
            out
        }
    }

    fn from_bytes<'a>(slice: impl Into<Cow<'a, [u8]>>, align: Align) -> Self {
        Self::alloc(FillBytes::Bytes(&*slice.into()), align)
            .unwrap()
    }

    fn zeroed(size: Size, align: Align) -> Option<Self> {
        Self::alloc(FillBytes::Zero(size), align)
    }
}