use super::*;

#[test]
fn test_errors_do_not_crash() {
    use std::path::Path;

    if !cfg!(unix) {
        return;
    }

    // Open /dev/null as a library to get an error, and make sure
    // that only causes an error, and not a crash.
    let path = Path::new("/dev/null");
    match DynamicLibrary::open(&path) {
        Err(_) => {}
        Ok(_) => panic!("Successfully opened the empty library."),
    }
}
