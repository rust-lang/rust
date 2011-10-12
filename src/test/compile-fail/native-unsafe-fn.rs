// -*- rust -*-
// error-pattern: unsafe functions can only be called

native "cdecl" mod test {
    unsafe fn free();
}

fn main() {
    let x = test::free;
}


