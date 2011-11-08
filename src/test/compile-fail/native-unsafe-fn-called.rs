// -*- rust -*-
// error-pattern: safe function calls function marked unsafe
native "c-stack-cdecl" mod test {
    unsafe fn free();
}

fn main() {
    test::free();
}

