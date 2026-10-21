//@only-target: windows # this directly tests windows-only functions
//@revisions: with_exit without_exit in_thread
//@run-native

use std::ffi::c_void;
use std::ptr;

pub type BOOL = i32;
pub const FALSE: BOOL = 0i32;
pub const TRUE: BOOL = 1i32;

extern "system" {
    fn TlsAlloc() -> u32;
    fn TlsSetValue(key: u32, val: *mut c_void) -> BOOL;
    fn TlsGetValue(key: u32) -> *mut c_void;
    fn TlsFree(key: u32) -> BOOL;

    fn FlsAlloc(lpcallback: Option<unsafe extern "system" fn(lpflsdata: *mut c_void)>) -> u32;
    fn FlsSetValue(key: u32, val: *mut c_void) -> BOOL;
    fn FlsGetValue(key: u32) -> *mut c_void;
    fn FlsFree(key: u32) -> BOOL;

    fn IsThreadAFiber() -> BOOL;
}

extern "system" fn dtor_unreachable(_val: *mut c_void) {
    unreachable!()
}

// Printing without touching the Rust runtime.
fn print_safe(msg: &str) {
    use std::io::Write;
    use std::os::windows::io::FromRawHandle;
    use std::{ffi, fs, mem};

    pub const STD_OUTPUT_HANDLE: u32 = 4294967285;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetStdHandle(which: u32) -> *mut ffi::c_void;
    }

    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut handle = mem::ManuallyDrop::new(fs::File::from_raw_handle(handle));
        handle.write_all(msg.as_bytes()).unwrap();
    }
}

fn fls_0_zero_value_does_not_run() {
    let key = unsafe { FlsAlloc(Some(dtor_unreachable)) };

    assert_eq!(unsafe { FlsSetValue(key, ptr::without_provenance_mut(1)) }, TRUE);
    assert_eq!(unsafe { FlsGetValue(key).addr() }, 1);
    assert_eq!(unsafe { FlsSetValue(key, ptr::without_provenance_mut(0)) }, TRUE);
}

fn fls_1_dtor_simple() {
    extern "system" fn dtor(val: *mut c_void) {
        assert!(!val.is_null());
        print_safe("fls_1_dtor_simple\n");

        // Keys are freed in-order. Without a dtor, the early key's value is not zeroed out.
        let early_key = val as u32;
        assert_eq!(unsafe { FlsGetValue(early_key).addr() }, 1);
    }

    let early_key = unsafe { FlsAlloc(None) };
    let later_key = unsafe { FlsAlloc(Some(dtor)) };

    assert_eq!(unsafe { FlsSetValue(later_key, ptr::without_provenance_mut(1)) }, TRUE);
    assert_eq!(unsafe { FlsGetValue(later_key).addr() }, 1);

    assert_eq!(unsafe { FlsSetValue(early_key, ptr::without_provenance_mut(1)) }, TRUE);
    // Will be used in the dtor to check early_key's value.
    assert_eq!(
        unsafe { FlsSetValue(later_key, ptr::without_provenance_mut(early_key as usize)) },
        TRUE
    );
}

fn fls_2_dtor_update_value_ignored() {
    extern "system" fn early_dtor(val: *mut c_void) {
        assert!(!val.is_null());
        print_safe("fls_2.1_early_dtor\n");
    }

    extern "system" fn later_dtor(val: *mut c_void) {
        print_safe("fls_2.2_later_dtor\n");

        // Updating a different fls slot's value doesn't cause their dtor to run, if it already did.
        let early_key = val as u32;

        // After the early key's dtor run, the key's value is zeroed out.
        assert_eq!(unsafe { FlsGetValue(early_key).addr() }, 0);

        assert_eq!(unsafe { FlsSetValue(early_key, ptr::without_provenance_mut(1)) }, TRUE);

        // Registering new fls slots in a dtor is always possible,
        // but their dtor might not get called.
        extern "system" fn new_in_dtor(_val: *mut c_void) {}
        let a_new_key_in_dtor = unsafe { FlsAlloc(Some(new_in_dtor)) };
        assert_eq!(unsafe { FlsSetValue(a_new_key_in_dtor, ptr::without_provenance_mut(1)) }, TRUE);
        assert_eq!(unsafe { FlsGetValue(a_new_key_in_dtor).addr() }, 1);
    }

    let early_key = unsafe { FlsAlloc(Some(early_dtor)) };
    let later_key = unsafe { FlsAlloc(Some(later_dtor)) };
    assert_eq!(unsafe { FlsSetValue(early_key, ptr::without_provenance_mut(1)) }, TRUE);
    assert_eq!(
        unsafe { FlsSetValue(later_key, ptr::without_provenance_mut(early_key as usize)) },
        TRUE
    );
}

fn fls_3_dtor_update_value_skipped_ignored() {
    extern "system" fn later_dtor(val: *mut c_void) {
        print_safe("fls_3_later_dtor\n");

        // Updating a different fls slot's value doesn't cause their dtor to run, if it was already skipped.
        let early_key = val as u32;
        assert_eq!(unsafe { FlsSetValue(early_key, ptr::without_provenance_mut(1)) }, TRUE);
    }

    let early_key = unsafe { FlsAlloc(Some(dtor_unreachable)) };
    let later_key = unsafe { FlsAlloc(Some(later_dtor)) };
    assert_eq!(unsafe { FlsSetValue(early_key, ptr::without_provenance_mut(0)) }, TRUE);
    assert_eq!(
        unsafe { FlsSetValue(later_key, ptr::without_provenance_mut(early_key as usize)) },
        TRUE
    );
}

fn fls_4_dtor_update_value_used() {
    extern "system" fn early_dtor(val: *mut c_void) {
        print_safe("fls_4.1_early_dtor\n");

        // Updating a different fls slot's value affect their dtor, if it hasn't yet run.
        let later_key = val as u32;
        assert_eq!(unsafe { FlsSetValue(later_key, ptr::without_provenance_mut(1)) }, TRUE);
    }

    extern "system" fn later_dtor(_val: *mut c_void) {
        print_safe("fls_4.2_later_dtor\n");
    }

    let early_key = unsafe { FlsAlloc(Some(early_dtor)) };
    let later_key = unsafe { FlsAlloc(Some(later_dtor)) };
    assert_eq!(
        unsafe { FlsSetValue(early_key, ptr::without_provenance_mut(later_key as usize)) },
        TRUE
    );

    // Setting to zero explicitly, dtor won't run unless `early_dtor` changes this value.
    assert_eq!(unsafe { FlsSetValue(later_key, ptr::without_provenance_mut(0)) }, TRUE);
}

fn fls_5_dtor_value() {
    extern "system" fn dtor(val: *mut c_void) {
        assert!(!val.is_null());
        print_safe("fls_5_dtor_value\n");

        // When the key's dtor run, the key's value equals the destructor argument.
        let key = val as u32;
        assert_eq!(unsafe { FlsGetValue(key) }, val);
    }

    let key = unsafe { FlsAlloc(Some(dtor)) };

    assert_eq!(unsafe { FlsSetValue(key, ptr::without_provenance_mut(key as usize)) }, TRUE);
}

fn fls_6_no_dtor_does_not_zero_value() {
    extern "system" fn later_dtor(val: *mut c_void) {
        print_safe("fls_6_later_dtor\n");

        // If an earlier fls slot doesn't have a dtor, the key's value is _not_ zeroed out.
        let early_key = val as u32;
        assert_ne!(unsafe { FlsGetValue(early_key).addr() }, 0);
    }

    let early_key = unsafe { FlsAlloc(None) };
    let later_key = unsafe { FlsAlloc(Some(later_dtor)) };
    assert_eq!(unsafe { FlsSetValue(early_key, ptr::without_provenance_mut(1)) }, TRUE);
    assert_eq!(
        unsafe { FlsSetValue(later_key, ptr::without_provenance_mut(early_key as usize)) },
        TRUE
    );
}

fn fls_7_free() {
    // This dtor is not expected to be called.
    extern "system" fn dtor_for_freed_key(_val: *mut c_void) {
        print_safe("dtor_for_freed_key\n");
    }
    let key_with_dtor = unsafe { FlsAlloc(Some(dtor_for_freed_key)) };
    assert_eq!(unsafe { FlsSetValue(key_with_dtor, ptr::without_provenance_mut(0)) }, TRUE);
    assert_eq!(unsafe { FlsFree(key_with_dtor) }, TRUE);

    let key_without_dtor = unsafe { FlsAlloc(None) };
    assert_eq!(unsafe { FlsSetValue(key_without_dtor, ptr::without_provenance_mut(1)) }, TRUE);
    assert_eq!(unsafe { FlsFree(key_without_dtor) }, TRUE);
}

fn fls() {
    assert_eq!(unsafe { IsThreadAFiber() }, FALSE);

    fls_0_zero_value_does_not_run();
    fls_1_dtor_simple();
    fls_2_dtor_update_value_ignored();
    fls_3_dtor_update_value_skipped_ignored();
    fls_4_dtor_update_value_used();
    fls_5_dtor_value();
    fls_6_no_dtor_does_not_zero_value();
    fls_7_free();
}

fn the_test() {
    let key = unsafe { TlsAlloc() };
    assert_eq!(unsafe { TlsSetValue(key, ptr::without_provenance_mut(1)) }, TRUE);
    assert_eq!(unsafe { TlsGetValue(key).addr() }, 1);
    assert_eq!(unsafe { TlsFree(key) }, TRUE);

    fls();
    print_safe("exiting thread\n");
}

fn main() {
    if cfg!(in_thread) {
        return std::thread::spawn(the_test).join().unwrap();
    }

    the_test();

    if cfg!(with_exit) {
        std::process::exit(0);
    }
}
