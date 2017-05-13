//! 32-bit specific Apple (ios/darwin) definitions

pub type c_long = i32;
pub type c_ulong = u32;

s! {
    pub struct pthread_attr_t {
        __sig: c_long,
        __opaque: [::c_char; 36]
    }
}

pub const __PTHREAD_MUTEX_SIZE__: usize = 40;
pub const __PTHREAD_COND_SIZE__: usize = 24;
pub const __PTHREAD_CONDATTR_SIZE__: usize = 4;
pub const __PTHREAD_RWLOCK_SIZE__: usize = 124;

pub const TIOCTIMESTAMP: ::c_ulong = 0x40087459;
pub const TIOCDCDTIMESTAMP: ::c_ulong = 0x40087458;
