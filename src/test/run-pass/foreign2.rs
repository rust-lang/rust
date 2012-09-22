#[abi = "cdecl"]
#[nolink]
extern mod bar {
    #[legacy_exports]; }

#[abi = "cdecl"]
#[nolink]
extern mod zed {
    #[legacy_exports]; }

#[abi = "cdecl"]
#[nolink]
extern mod libc {
    #[legacy_exports];
    fn write(fd: int, buf: *u8,
             count: core::libc::size_t) -> core::libc::ssize_t;
}

#[abi = "cdecl"]
#[nolink]
extern mod baz {
    #[legacy_exports]; }

fn main(args: ~[~str]) { }
