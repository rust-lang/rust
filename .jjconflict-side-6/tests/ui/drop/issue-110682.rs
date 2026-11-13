//@ build-pass
//@ compile-flags: -Zmir-opt-level=3

use std::fmt::Debug;
use std::mem::ManuallyDrop;
use std::ptr;

pub trait BitRegister {}

macro_rules! register {
    ($($t:ty),+ $(,)?) => { $(
        impl BitRegister for $t {
        }
    )* };
}

register!(u8, u16, u32);

pub trait BitStore: Sized + Debug {
    /// The register type that the implementor describes.
    type Mem: BitRegister + Into<Self>;
}

macro_rules! store {
    ($($t:ty),+ $(,)?) => { $(
        impl BitStore for $t {
            type Mem = Self;
        }
    )+ };
}

store!(u8, u16, u32,);

#[repr(C)]
pub struct BitVec<T>
where
    T: BitStore,
{
    /// Region pointer describing the live portion of the owned buffer.
    pointer: ptr::NonNull<T>,
    /// Allocated capacity, in elements `T`, of the owned buffer.
    capacity: usize,
}

impl<T> BitVec<T>
where
    T: BitStore,
{
    pub fn new() -> Self {
        let pointer = ptr::NonNull::<T>::new(ptr::null_mut()).unwrap();

        BitVec { pointer, capacity: 10 }
    }

    pub fn clear(&mut self) {
        unsafe {
            self.set_len(0);
        }
    }

    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {}

    fn with_vec<F, R>(&mut self, func: F) -> R
    where
        F: FnOnce(&mut ManuallyDrop<Vec<T::Mem>>) -> R,
    {
        let cap = self.capacity;
        let elts = 10;
        let mut vec = ManuallyDrop::new(unsafe { Vec::from_raw_parts(ptr::null_mut(), elts, cap) });
        let out = func(&mut vec);

        out
    }
}

impl<T> Drop for BitVec<T>
where
    T: BitStore,
{
    #[inline]
    fn drop(&mut self) {
        //  The buffer elements do not have destructors.
        self.clear();
        //  Run the `Vec` destructor to deällocate the buffer.
        self.with_vec(|vec| unsafe { ManuallyDrop::drop(vec) });
    }
}

fn main() {
    let bitvec = BitVec::<u32>::new();
}
