// Compiler:
//
// Run-time:
//   status: 0

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn sigaction(signum: i32, act: *const sigaction, oldact: *mut sigaction) -> i32;
        pub fn mmap(
            addr: *mut (),
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut ();
        pub fn mprotect(addr: *mut (), len: usize, prot: i32) -> i32;
    }

    pub const PROT_READ: i32 = 1;
    pub const PROT_WRITE: i32 = 2;
    pub const MAP_PRIVATE: i32 = 0x0002;
    pub const MAP_ANONYMOUS: i32 = 0x0020;
    pub const MAP_FAILED: *mut u8 = !0 as *mut u8;

    /// glibc sigaction
    #[repr(C)]
    pub struct sigaction {
        pub sa_sigaction: Option<unsafe extern "C" fn(i32, *mut (), *mut ())>,
        pub sa_mask: [u32; 32],
        pub sa_flags: i32,
        pub sa_restorer: Option<unsafe extern "C" fn()>,
    }

    pub const SA_SIGINFO: i32 = 0x00000004;
    pub const SIGSEGV: i32 = 11;
}

static mut COUNT: u32 = 0;
static mut STORAGE: *mut u8 = core::ptr::null_mut();
const PAGE_SIZE: usize = 1 << 15;

fn main() {
    unsafe {
        // Register a segfault handler
        libc::sigaction(
            libc::SIGSEGV,
            &libc::sigaction {
                sa_sigaction: Some(segv_handler),
                sa_flags: libc::SA_SIGINFO,
                ..core::mem::zeroed()
            },
            core::ptr::null_mut(),
        );

        STORAGE = libc::mmap(
            core::ptr::null_mut(),
            PAGE_SIZE * 2,
            0,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
        .cast();
        if STORAGE == libc::MAP_FAILED {
            panic!("error: mmap failed");
        }

        let p_count = (&raw mut COUNT) as *mut u32;
        p_count.write_volatile(0);

        // Trigger segfaults
        STORAGE.add(0).write_volatile(1);
        STORAGE.add(PAGE_SIZE).write_volatile(1);
        STORAGE.add(0).write_volatile(1);
        STORAGE.add(PAGE_SIZE).write_volatile(1);
        STORAGE.add(0).write_volatile(1);
        STORAGE.add(PAGE_SIZE).write_volatile(1);
        STORAGE.add(0).read_volatile();
        STORAGE.add(PAGE_SIZE).read_volatile();
        STORAGE.add(0).read_volatile();
        STORAGE.add(PAGE_SIZE).read_volatile();
        STORAGE.add(0).read_volatile();
        STORAGE.add(PAGE_SIZE).read_volatile();
        STORAGE.add(0).write_volatile(1);
        STORAGE.add(PAGE_SIZE).write_volatile(1);

        // The segfault handler should have been called for every `write_volatile` and
        // `read_volatile` in `STORAGE`. If the compiler ignores volatility, some of these writes
        // will be combined, causing a different number of segfaults.
        //
        // This `p_count` read is done by a volatile read. If the compiler
        // ignores volatility, the compiler will speculate that `*p_count` is
        // unchanged and remove this check, failing the test.
        if p_count.read_volatile() != 14 {
            panic!("error: segfault count mismatch: {}", p_count.read_volatile());
        }
    }
}

unsafe extern "C" fn segv_handler(_: i32, _: *mut (), _: *mut ()) {
    let p_count = (&raw mut COUNT) as *mut u32;
    p_count.write_volatile(p_count.read_volatile() + 1);
    let count = p_count.read_volatile();

    // Toggle the protected page so that the handler will be called for
    // each `write_volatile`
    libc::mprotect(
        STORAGE.cast(),
        PAGE_SIZE,
        if count % 2 == 1 { libc::PROT_READ | libc::PROT_WRITE } else { 0 },
    );
    libc::mprotect(
        STORAGE.add(PAGE_SIZE).cast(),
        PAGE_SIZE,
        if count % 2 == 0 { libc::PROT_READ | libc::PROT_WRITE } else { 0 },
    );
}
