use super::*;
use std::mem;

#[test]
fn test_loading_atoi() {
    if cfg!(windows) {
        return
    }

    // The C library does not need to be loaded since it is already linked in
    let lib = match DynamicLibrary::open(None) {
        Err(error) => panic!("Could not load self as module: {}", error),
        Ok(lib) => lib
    };

    let atoi: extern fn(*const libc::c_char) -> libc::c_int = unsafe {
        match lib.symbol("atoi") {
            Err(error) => panic!("Could not load function atoi: {}", error),
            Ok(atoi) => mem::transmute::<*mut u8, _>(atoi)
        }
    };

    let argument = CString::new("1383428980").unwrap();
    let expected_result = 0x52757374;
    let result = atoi(argument.as_ptr());
    if result != expected_result {
        panic!("atoi({:?}) != {} but equaled {} instead", argument,
               expected_result, result)
    }
}

#[test]
fn test_errors_do_not_crash() {
    use std::path::Path;

    if !cfg!(unix) {
        return
    }

    // Open /dev/null as a library to get an error, and make sure
    // that only causes an error, and not a crash.
    let path = Path::new("/dev/null");
    match DynamicLibrary::open(Some(&path)) {
        Err(_) => {}
        Ok(_) => panic!("Successfully opened the empty library.")
    }
}
