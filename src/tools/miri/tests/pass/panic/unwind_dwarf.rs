//@only-target-linux
#![feature(core_intrinsics, panic_unwind, rustc_attrs)]
#![allow(internal_features)]

//! Unwinding using `_Unwind_RaiseException`

extern crate unwind as uw;

use std::any::Any;
use std::ptr;

#[repr(C)]
struct Exception {
    _uwe: uw::_Unwind_Exception,
    cause: Box<dyn Any + Send>,
}

pub fn panic(data: Box<dyn Any + Send>) -> u32 {
    let exception = Box::new(Exception {
        _uwe: uw::_Unwind_Exception {
            exception_class: rust_exception_class(),
            exception_cleanup,
            private: [core::ptr::null(); uw::unwinder_private_data_size],
        },
        cause: data,
    });
    let exception_param = Box::into_raw(exception) as *mut uw::_Unwind_Exception;
    return unsafe { uw::_Unwind_RaiseException(exception_param) as u32 };

    extern "C" fn exception_cleanup(
        _unwind_code: uw::_Unwind_Reason_Code,
        _exception: *mut uw::_Unwind_Exception,
    ) {
        std::process::abort();
    }
}

pub unsafe fn rust_panic_cleanup(ptr: *mut u8) -> Box<dyn Any + Send> {
    let exception = ptr as *mut uw::_Unwind_Exception;
    if (*exception).exception_class != rust_exception_class() {
        std::process::abort();
    }

    let exception = exception.cast::<Exception>();

    let exception = Box::from_raw(exception as *mut Exception);
    exception.cause
}

fn rust_exception_class() -> uw::_Unwind_Exception_Class {
    // M O Z \0  R U S T -- vendor, language
    0x4d4f5a_00_52555354
}

pub fn catch_unwind<R, F: FnOnce() -> R>(f: F) -> Result<R, Box<dyn Any + Send>> {
    struct Data<F, R> {
        f: Option<F>,
        r: Option<R>,
        p: Option<Box<dyn Any + Send>>,
    }

    let mut data = Data { f: Some(f), r: None, p: None };

    let data_ptr = ptr::addr_of_mut!(data) as *mut u8;
    unsafe {
        return if std::intrinsics::r#try(do_call::<F, R>, data_ptr, do_catch::<F, R>) == 0 {
            Ok(data.r.take().unwrap())
        } else {
            Err(data.p.take().unwrap())
        };
    }

    fn do_call<F: FnOnce() -> R, R>(data: *mut u8) {
        unsafe {
            let data = &mut *data.cast::<Data<F, R>>();
            let f = data.f.take().unwrap();
            data.r = Some(f());
        }
    }

    #[rustc_nounwind]
    fn do_catch<F: FnOnce() -> R, R>(data: *mut u8, payload: *mut u8) {
        unsafe {
            let obj = rust_panic_cleanup(payload);
            (*data.cast::<Data<F, R>>()).p = Some(obj);
        }
    }
}

fn main() {
    assert_eq!(
        catch_unwind(|| panic(Box::new(42))).unwrap_err().downcast::<i32>().unwrap(),
        Box::new(42)
    );
}
