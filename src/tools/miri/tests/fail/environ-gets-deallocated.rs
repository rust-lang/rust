//@ignore-target-windows: Windows does not have a global environ list that the program can access directly

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
fn get_environ() -> *const *const u8 {
    extern "C" {
        static mut environ: *const *const u8;
    }
    unsafe { environ }
}

#[cfg(target_os = "macos")]
fn get_environ() -> *const *const u8 {
    extern "C" {
        fn _NSGetEnviron() -> *mut *const *const u8;
    }
    unsafe { *_NSGetEnviron() }
}

fn main() {
    let pointer = get_environ();
    let _x = unsafe { *pointer };
    std::env::set_var("FOO", "BAR");
    let _y = unsafe { *pointer }; //~ ERROR: has been freed
}
