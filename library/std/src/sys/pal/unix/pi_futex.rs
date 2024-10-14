#[cfg(any(target_os = "linux", target_os = "android"))]
mod linux {
    use crate::ops::Deref;
    use crate::sync::atomic::AtomicU32;
    use crate::sys::cvt;
    use crate::{io, ptr};

    pub type State = u32;

    pub struct Futex(AtomicU32);

    impl Futex {
        pub const fn new() -> Futex {
            Futex(AtomicU32::new(0))
        }
    }

    impl Deref for Futex {
        type Target = AtomicU32;
        fn deref(&self) -> &AtomicU32 {
            &self.0
        }
    }

    pub const fn unlocked() -> State {
        0
    }

    pub fn locked() -> State {
        (unsafe { libc::gettid() }) as _
    }

    pub fn is_contended(futex_val: State) -> bool {
        (futex_val & libc::FUTEX_WAITERS) != 0
    }

    pub fn is_owned_died(futex_val: State) -> bool {
        (futex_val & libc::FUTEX_OWNER_DIED) != 0
    }

    pub fn futex_lock(futex: &Futex) -> io::Result<()> {
        loop {
            match cvt(unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    ptr::from_ref(futex.deref()),
                    libc::FUTEX_LOCK_PI | libc::FUTEX_PRIVATE_FLAG,
                    0,
                    ptr::null::<u32>(),
                    // remaining args are unused
                )
            }) {
                Ok(_) => return Ok(()),
                Err(e) if e.raw_os_error() == Some(libc::EINTR) => continue,
                Err(e) => return Err(e),
            }
        }
    }

    pub fn futex_unlock(futex: &Futex) -> io::Result<()> {
        cvt(unsafe {
            libc::syscall(
                libc::SYS_futex,
                ptr::from_ref(futex.deref()),
                libc::FUTEX_UNLOCK_PI | libc::FUTEX_PRIVATE_FLAG,
                // remaining args are unused
            )
        })
        .map(|_| ())
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub use linux::*;

#[cfg(target_os = "freebsd")]
mod freebsd {
    use crate::mem::transmute;
    use crate::ops::Deref;
    use crate::sync::atomic::AtomicU32;
    use crate::sys::cvt;
    use crate::{io, ptr};

    pub type State = u32;

    #[repr(C)]
    pub struct umutex {
        m_owner: libc::lwpid_t,
        m_flags: u32,
        m_ceilings: [u32; 2],
        m_rb_link: libc::uintptr_t,
        #[cfg(target_pointer_width = "32")]
        m_pad: u32,
        m_spare: [u32; 2],
    }

    pub struct Futex(umutex);

    impl Futex {
        pub const fn new() -> Futex {
            Futex(umutex {
                m_owner: 0,
                m_flags: UMUTEX_PRIO_INHERIT,
                m_ceilings: [0, 0],
                m_rb_link: 0,
                #[cfg(target_pointer_width = "32")]
                m_pad: 0,
                m_spare: [0, 0],
            })
        }
    }

    impl Deref for Futex {
        type Target = AtomicU32;
        fn deref(&self) -> &AtomicU32 {
            unsafe { transmute(&self.0.m_owner) }
        }
    }

    const UMUTEX_PRIO_INHERIT: u32 = 0x0004;
    const UMUTEX_CONTESTED: u32 = 0x80000000;

    pub const fn unlocked() -> State {
        0
    }

    pub fn locked() -> State {
        let mut tid: libc::c_long = 0;
        let _ = unsafe { libc::thr_self(ptr::from_mut(&mut tid)) };
        tid as _
    }

    pub fn is_contended(futex_val: State) -> bool {
        (futex_val & UMUTEX_CONTESTED) != 0
    }

    pub fn is_owned_died(futex_val: State) -> bool {
        // never happens for non-robust mutex
        let _ = futex_val;
        false
    }

    pub fn futex_lock(futex: &Futex) -> io::Result<()> {
        cvt(unsafe {
            libc::_umtx_op(
                ptr::from_ref(futex.deref()) as _,
                libc::UMTX_OP_MUTEX_LOCK,
                0,
                ptr::null_mut::<libc::c_void>(),
                ptr::null_mut::<libc::c_void>(),
            )
        })
        .map(|_| ())
    }

    pub fn futex_unlock(futex: &Futex) -> io::Result<()> {
        cvt(unsafe {
            libc::_umtx_op(
                ptr::from_ref(futex.deref()) as _,
                libc::UMTX_OP_MUTEX_UNLOCK,
                0,
                ptr::null_mut::<libc::c_void>(),
                ptr::null_mut::<libc::c_void>(),
            )
        })
        .map(|_| ())
    }
}

#[cfg(target_os = "freebsd")]
pub use freebsd::*;
