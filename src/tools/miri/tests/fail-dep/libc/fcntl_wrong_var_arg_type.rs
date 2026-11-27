//@ignore-target: windows # testing POSIX API
//@normalize-stderr-test: "to have type `(u32|i32)`" -> "to have type `$$TYPE`"

fn main() {
    let _fd = unsafe {
        libc::fcntl(0, libc::F_DUPFD, /* should be i32 */ u32::MAX)
        //~^ ERROR: /has value `\d+_u32` which cannot be represented in expected type `i32`/
    };
}
