use std::borrow::Cow;
use std::fmt;
use std::hash::Hash;
use std::mem::size_of;
use std::ops::Deref;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_serialize::{Decodable, Encodable};
use rustc_span::DUMMY_SP;

use super::{InterpError, InterpResult, ResourceExhaustionInfo};
use crate::ty;
use crate::ty::codec::{TyDecoder, TyEncoder};

/// Equivalent to `Box<[u8]>` in behaviour, but slices that have fewer (or the same) bytes as
/// a pointer on the host system will encode the bytes directly in the pointer instead of
/// allocating on the heap.
pub struct SmallSlice {
    // INVARIANT: we use the `small` variant if `len <= SMALL_CAPACITY` and the `ptr` variant
    // otherwise. Even though larger lengths could be encoded via the `ptr` variant, we have
    // no way to know what variant is being used beyond the length.
    data: SmallPointer,
    len: usize,
}

impl Drop for SmallSlice {
    fn drop(&mut self) {
        if self.len > Self::SMALL_CAPACITY {
            unsafe {
                Box::from_raw(std::ptr::slice_from_raw_parts_mut(self.data.ptr, self.len));
            }
        }
    }
}

// SAFETY: `SmallSlice` is equivalent to `Box<[u8]>`. Since it doesn't actually allow mutating
// via immutable methods, we can safely share it across threads.
unsafe impl Sync for SmallSlice {}
// SAFETY: `SmallSlice` is equivalent to `Box<[u8]>`. Since it doesn't contain anything that
// msut stay on one thread, we can safely move it to another thread.
unsafe impl Send for SmallSlice {}

impl Clone for SmallSlice {
    fn clone(&self) -> Self {
        if self.len <= Self::SMALL_CAPACITY {
            let &Self { data, len } = self;
            Self { data, len }
        } else {
            Self {
                data: SmallPointer { ptr: Box::<[u8]>::into_raw(Box::from(self.deref())) as _ },
                len: self.len,
            }
        }
    }
}

#[derive(Copy, Clone)]
union SmallPointer {
    small: [u8; SmallSlice::SMALL_CAPACITY],
    ptr: *mut u8,
}

impl SmallSlice {
    const SMALL_CAPACITY: usize = size_of::<*mut u8>();

    pub fn zeroed(len: usize, panic_on_fail: bool) -> InterpResult<'static, Self> {
        if len <= Self::SMALL_CAPACITY {
            return Ok(SmallSlice {
                len,
                data: SmallPointer { small: [0; SmallSlice::SMALL_CAPACITY] },
            });
        }
        let bytes = Box::<[u8]>::try_new_zeroed_slice(len).map_err(|_| {
            // This results in an error that can happen non-deterministically, since the memory
            // available to the compiler can change between runs. Normally queries are always
            // deterministic. However, we can be non-determinstic here because all uses of const
            // evaluation (including ConstProp!) will make compilation fail (via hard error
            // or ICE) upon encountering a `MemoryExhausted` error.
            if panic_on_fail {
                panic!("Allocation::uninit called with panic_on_fail had allocation failure")
            }
            ty::tls::with(|tcx| {
                tcx.sess.delay_span_bug(DUMMY_SP, "exhausted memory during interpreation")
            });
            InterpError::ResourceExhaustion(ResourceExhaustionInfo::MemoryExhausted)
        })?;
        // SAFETY: the box was zero-allocated, which is a valid initial value for Box<[u8]>
        Ok(Self {
            len,
            data: SmallPointer { ptr: Box::<[u8]>::into_raw(unsafe { bytes.assume_init() }) as _ },
        })
    }
}

impl<'a> From<Cow<'a, [u8]>> for SmallSlice {
    fn from(cow: Cow<'a, [u8]>) -> Self {
        Self {
            len: cow.len(),
            data: if cow.len() <= SmallSlice::SMALL_CAPACITY {
                let mut small = [0; SmallSlice::SMALL_CAPACITY];
                small[..cow.len()].copy_from_slice(&cow);
                SmallPointer { small }
            } else {
                SmallPointer { ptr: Box::<[u8]>::into_raw(Box::<[u8]>::from(cow)) as _ }
            },
        }
    }
}

impl Deref for SmallSlice {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        if self.len <= Self::SMALL_CAPACITY {
            // SAFETY: the union is guaranteed initialized at the `small` variant, since
            // short slices are always encoded as the small variant.
            unsafe { &self.data.small[..self.len] }
        } else {
            unsafe { std::slice::from_raw_parts(self.data.ptr, self.len) }
        }
    }
}

impl std::ops::DerefMut for SmallSlice {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.len <= Self::SMALL_CAPACITY {
            // SAFETY: the union is guaranteed initialized at the `small` variant, since
            // short slices are always encoded as the small variant.
            unsafe { &mut self.data.small[..self.len] }
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.data.ptr, self.len) }
        }
    }
}

impl fmt::Debug for SmallSlice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}

impl Eq for SmallSlice {}
impl PartialEq for SmallSlice {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl Ord for SmallSlice {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.deref().cmp(other.deref())
    }
}
impl PartialOrd for SmallSlice {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.deref().partial_cmp(other.deref())
    }
}

impl Hash for SmallSlice {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.deref().hash(state)
    }
}

impl<CTX> HashStable<CTX> for SmallSlice {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.deref().hash_stable(hcx, hasher)
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for SmallSlice {
    fn decode(d: &mut D) -> Self {
        Self::from(Cow::Owned(<Vec<u8> as Decodable<D>>::decode(d)))
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for SmallSlice {
    fn encode(&self, e: &mut E) -> Result<(), E::Error> {
        self.deref().encode(e)
    }
}
