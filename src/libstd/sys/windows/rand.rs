use io;
use mem;
use rand::Rng;
use libc::types::os::arch::extra::{LONG_PTR};
use libc::{DWORD, BYTE, LPCSTR, BOOL};
use sys::error::{self, Result};

type HCRYPTPROV = LONG_PTR;

/// A random number generator that retrieves randomness straight from
/// the operating system. Platform sources:
///
/// - Unix-like systems (Linux, Android, Mac OSX): read directly from
///   `/dev/urandom`, or from `getrandom(2)` system call if available.
/// - Windows: calls `CryptGenRandom`, using the default cryptographic
///   service provider with the `PROV_RSA_FULL` type.
/// - iOS: calls SecRandomCopyBytes as /dev/(u)random is sandboxed.
///
/// This does not block.
pub struct OsRng {
    hcryptprov: HCRYPTPROV
}

const PROV_RSA_FULL: DWORD = 1;
const CRYPT_SILENT: DWORD = 64;
const CRYPT_VERIFYCONTEXT: DWORD = 0xF0000000;

#[allow(non_snake_case)]
#[link(name = "advapi32")]
extern "system" {
    fn CryptAcquireContextA(phProv: *mut HCRYPTPROV,
                            pszContainer: LPCSTR,
                            pszProvider: LPCSTR,
                            dwProvType: DWORD,
                            dwFlags: DWORD) -> BOOL;
    fn CryptGenRandom(hProv: HCRYPTPROV,
                      dwLen: DWORD,
                      pbBuffer: *mut BYTE) -> BOOL;
    fn CryptReleaseContext(hProv: HCRYPTPROV, dwFlags: DWORD) -> BOOL;
}

impl OsRng {
    /// Create a new `OsRng`.
    pub fn new() -> Result<OsRng> {
        let mut hcp = 0;
        let ret = unsafe {
            CryptAcquireContextA(&mut hcp, 0 as LPCSTR, 0 as LPCSTR,
                                 PROV_RSA_FULL,
                                 CRYPT_VERIFYCONTEXT | CRYPT_SILENT)
        };

        if ret == 0 {
            error::expect_last_result()
        } else {
            Ok(OsRng { hcryptprov: hcp })
        }
    }
}

impl Rng for OsRng {
    fn next_u32(&mut self) -> u32 {
        let mut v = [0; 4];
        self.fill_bytes(&mut v);
        unsafe { mem::transmute(v) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut v = [0; 8];
        self.fill_bytes(&mut v);
        unsafe { mem::transmute(v) }
    }
    fn fill_bytes(&mut self, v: &mut [u8]) {
        let ret = unsafe {
            CryptGenRandom(self.hcryptprov, v.len() as DWORD,
                           v.as_mut_ptr())
        };
        if ret == 0 {
            panic!("couldn't generate random bytes: {}",
                   io::Error::last_os_error());
        }
    }
}

impl Drop for OsRng {
    fn drop(&mut self) {
        let ret = unsafe {
            CryptReleaseContext(self.hcryptprov, 0)
        };
        if ret == 0 {
            panic!("couldn't release context: {}",
                   io::Error::last_os_error());
        }
    }
}
