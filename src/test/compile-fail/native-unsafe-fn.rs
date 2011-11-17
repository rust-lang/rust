// -*- rust -*-
// error-pattern: unsafe functions can only be called

#[abi = "cdecl"]
native mod test {
    unsafe fn free();
}

fn main() {
    let x = test::free;
}


