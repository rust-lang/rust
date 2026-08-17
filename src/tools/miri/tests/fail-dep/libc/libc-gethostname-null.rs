//@ignore-target: windows # No libc

fn main() {
    // Some libc implementations dereference `name` in userspace. For example, glibc obtains the
    // hostname with `uname` and then copies it into `name` with `memcpy`, so a null pointer is UB:
    // <https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/posix/gethostname.c;h=b17671373ca1ac61e7346ae0dcdda22eff1d7fcd#l36>
    unsafe {
        libc::gethostname(std::ptr::null_mut(), 5); //~ERROR: null pointer
    }
}
