//@compile-flags: -Zmiri-disable-isolation
//@only-target: linux android illumos
//@ignore-host: windows

fn main() {
    let _ = match std::fs::File::open("/proc/doesnotexist ") {
        Ok(_f) => {}
        Err(_msg) => {}
    };
    ();
}
