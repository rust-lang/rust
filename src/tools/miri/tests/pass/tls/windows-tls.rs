//@only-target: windows # this directly tests windows-only functions

use std::ffi::c_void;
use std::{ptr, thread};

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

    fn IsThreadAFiber() -> BOOL;
}

extern "system" fn dtor_unreachable(_val: *mut c_void) {
    unreachable!()
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
        println!("fls_1_dtor_simple");
        
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
    assert_eq!(unsafe { FlsSetValue(later_key, ptr::without_provenance_mut(early_key as usize)) }, TRUE);
}

fn fls_2_dtor_update_value_ignored() {
    extern "system" fn early_dtor(val: *mut c_void) {
        assert!(!val.is_null());
        println!("fls_2.1_early_dtor");
    }

    extern "system" fn later_dtor(val: *mut c_void) {
        println!("fls_2.2_later_dtor");

        // Updating a different fls slot's value doesn't cause their dtor to run, if it already did.
        let early_key = val as u32;
        
        // After the early key's dtor run, the key's value is zeroed out.
        assert_eq!(
            unsafe { FlsGetValue(early_key).addr() },
            0
        );

        assert_eq!(
            unsafe { FlsSetValue(early_key, ptr::without_provenance_mut(1)) },
            TRUE
        );

        // Registering new fls slots doesn't cause their dtor to run.
        let a_new_key_in_dtor = unsafe { FlsAlloc(Some(dtor_unreachable)) };
        assert_eq!(unsafe { FlsSetValue(a_new_key_in_dtor, ptr::without_provenance_mut(1)) }, TRUE);
    }

    let early_key = unsafe { FlsAlloc(Some(early_dtor)) };
    let later_key = unsafe { FlsAlloc(Some(later_dtor)) };
    assert_eq!(unsafe { FlsSetValue(early_key, ptr::without_provenance_mut(1)) }, TRUE);
    assert_eq!(unsafe { FlsSetValue(later_key, ptr::without_provenance_mut(early_key as usize)) }, TRUE);
}

fn fls_3_dtor_update_value_skipped_ignored() {
    extern "system" fn later_dtor(val: *mut c_void) {
        println!("fls_3_later_dtor");

        // Updating a different fls slot's value doesn't cause their dtor to run, if it was already skipped.
        let early_key = val as u32;
        assert_eq!(
            unsafe { FlsSetValue(early_key, ptr::without_provenance_mut(1)) },
            TRUE
        );
    }

    let early_key = unsafe { FlsAlloc(Some(dtor_unreachable)) };
    let later_key = unsafe { FlsAlloc(Some(later_dtor)) };
    assert_eq!(unsafe { FlsSetValue(early_key, ptr::without_provenance_mut(0)) }, TRUE);
    assert_eq!(unsafe { FlsSetValue(later_key, ptr::without_provenance_mut(early_key as usize)) }, TRUE);
}

fn fls_4_dtor_update_value_used() {
    extern "system" fn early_dtor(val: *mut c_void) {
        println!("fls_4.1_early_dtor");

        // Updating a different fls slot's value affect their dtor, if it hasn't yet run.
        let later_key = val as u32;
        assert_eq!(unsafe { FlsSetValue(later_key, ptr::without_provenance_mut(1)) }, TRUE);
    }

    extern "system" fn later_dtor(_val: *mut c_void) {
        println!("fls_4.2_later_dtor");
    }

    let early_key = unsafe { FlsAlloc(Some(early_dtor)) };
    let later_key = unsafe { FlsAlloc(Some(later_dtor)) };
    assert_eq!(unsafe { FlsSetValue(early_key, ptr::without_provenance_mut(later_key as usize)) }, TRUE);

    // Setting to zero explicitly, dtor won't rub unless `early_dtor` changes this value.
    assert_eq!(unsafe { FlsSetValue(later_key, ptr::without_provenance_mut(0)) }, TRUE);
}

fn fls_5_dtor_value() {
    extern "system" fn dtor(val: *mut c_void) {
        assert!(!val.is_null());
        println!("fls_5_dtor_value");
        
        // When the key's dtor run, the key's value equals the destructor argument.
        let key = val as u32;
        assert_eq!(unsafe { FlsGetValue(key) }, val);
    }

    let key = unsafe { FlsAlloc(Some(dtor)) };

    assert_eq!(unsafe { FlsSetValue(key, ptr::without_provenance_mut(key as usize)) }, TRUE);
}

fn fls() {
    assert_eq!(unsafe { IsThreadAFiber() }, FALSE);

    fls_0_zero_value_does_not_run();
    fls_1_dtor_simple();
    fls_2_dtor_update_value_ignored();
    fls_3_dtor_update_value_skipped_ignored();
    fls_4_dtor_update_value_used();
    fls_5_dtor_value();
}

fn main() {
    let key = unsafe { TlsAlloc() };
    assert_eq!(unsafe { TlsSetValue(key, ptr::without_provenance_mut(1)) }, TRUE);
    assert_eq!(unsafe { TlsGetValue(key).addr() }, 1);
    assert_eq!(unsafe { TlsFree(key) }, TRUE);

    thread::spawn(|| {
        fls();
        println!("exiting thread");
    })
    .join()
    .unwrap();
}
