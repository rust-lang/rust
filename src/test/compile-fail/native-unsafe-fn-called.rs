// -*- rust -*-
// error-pattern: safe function calls function marked unsafe
#[abi = "cdecl"]
native mod test {
    unsafe fn free();
}

fn main() {
    test::free();
}

