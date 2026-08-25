//@ignore-target: windows # No libc

fn main() {
    // MacOS unconditionally writes to `name[namelen - 1]`, so passing zero for `namelen`
    // causes an out-of-bounds access. Miri requires a non-zero length on every target:
    // <https://github.com/apple-oss-distributions/Libc/blob/main/gen/FreeBSD/gethostname.c#L48-L60>
    let mut name = 0u8;
    unsafe {
        libc::gethostname((&mut name as *mut u8).cast(), 0); //~ERROR: length of zero
    }
}
