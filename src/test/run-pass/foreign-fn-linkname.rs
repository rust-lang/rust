extern mod std;


#[nolink]
#[abi = "cdecl"]
extern mod libc {
    #[link_name = "strlen"]
    fn my_strlen(str: *u8) -> uint;
}

fn strlen(str: ~str) -> uint unsafe {
    // C string is terminated with a zero
    let bytes = str::to_bytes(str) + ~[0u8];
    return libc::my_strlen(vec::unsafe::to_ptr(bytes));
}

fn main() {
    let len = strlen(~"Rust");
    assert(len == 4u);
}
