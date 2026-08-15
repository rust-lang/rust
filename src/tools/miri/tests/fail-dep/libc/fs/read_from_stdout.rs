//@compile-flags: -Zmiri-disable-isolation
//@ignore-target: windows # No libc IO on Windows

fn main() -> std::io::Result<()> {
    let mut bytes = [0u8; 512];
    unsafe {
        libc::read(1, bytes.as_mut_ptr() as *mut libc::c_void, 512); //~ ERROR: cannot read from stdout
    }
    Ok(())
}
