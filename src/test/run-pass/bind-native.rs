/*
Can we bind native things?
*/

#[abi = "cdecl"]
native mod rustrt {
    fn rand_new() -> *libc::c_void;
}

fn main() { bind rustrt::rand_new(); }
