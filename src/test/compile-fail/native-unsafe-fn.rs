// -*- rust -*-
// error-pattern: unsafe functions can only be called

native "c-stack-cdecl" mod test {
    unsafe fn free();
}

fn main() {
    let x = test::free;
}


