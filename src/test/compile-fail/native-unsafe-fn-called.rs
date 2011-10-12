// -*- rust -*-
// error-pattern: safe function calls function marked unsafe

native "cdecl" mod test {
    unsafe fn free();
}

fn main() {
    test::free();
}

