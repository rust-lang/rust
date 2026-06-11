use std::fmt;
use std::mem::MaybeUninit;
use std::ptr;
use std::error::Error;

/// Custom error type for Vec operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VecError {
    /// Attempted to access uninitialized data.
    UninitializedData,
    /// Capacity overflow.
    CapacityOverflow,
    /// Invalid capacity value.
    InvalidCapacity(usize),
}

impl fmt::Display for VecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VecError::UninitializedData => write!(f, "attempted to access uninitialized data"),
            VecError::CapacityOverflow => write!(f, "capacity overflow"),
            VecError::InvalidCapacity(cap) => write!(f, "invalid capacity: {}", cap),
        }
    }
}

impl Error for VecError {}

/// A vector-like structure with a fixed-size backing array.
///
/// # Fields
/// * `cap` - The current capacity of the vector.
/// * `data` - Uninitialized memory for storing up to 2 `u64` values.
///
/// # Safety
/// The `data` field is intentionally left uninitialized until explicitly
/// initialized via the provided methods.
#[derive(Debug)]
pub struct Vec {
    cap: usize,
    data: MaybeUninit<[u64; 2]>,
}

impl Vec {
    /// Maximum allowed capacity.
    pub const MAX_CAPACITY: usize = 2;

    /// Creates a new `Vec` with zero capacity and uninitialized data.
    ///
    /// # Returns
    /// A `Vec` instance with `cap = 0` and uninitialized `data`.
    ///
    /// # Performance
    /// This implementation uses `unsafe` pointer writes to ensure only the
    /// `cap` field is initialized, avoiding unnecessary `memset` calls.
    #[inline]
    pub fn new() -> Self {
        unsafe {
            let mut result = MaybeUninit::<Vec>::uninit();
            let ptr = result.as_mut_ptr();
            ptr::write(&mut (*ptr).cap, 0_usize);
            result.assume_init()
        }
    }

    /// Returns the current capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap