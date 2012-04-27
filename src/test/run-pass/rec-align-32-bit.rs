// xfail-pretty
// xfail-win32
// Issue #2303

// This is the type with the questionable alignment
type inner = {
    c64: u64
};

// This is the type that contains the type with the
// questionable alignment, for testing
type outer = {
    c8: u8,
    t: inner
};

#[cfg(target_arch = "x86")]
fn main() {

    let x = {c8: 22u8, t: {c64: 44u64}};

    // Send it through the shape code
    let y = #fmt["%?", x];

    #debug("align inner = %?", sys::align_of::<inner>()); // 8
    #debug("size outer = %?", sys::size_of::<outer>());   // 12
    #debug("y = %s", y);                                  // (22, (0))

    // per clang/gcc the alignment of `inner` is 4 on x86.
    // we say it's 8
    //assert sys::align_of::<inner>() == 4u; // fails

    // per clang/gcc the size of `outer` should be 12
    // because `inner`s alignment was 4.
    // LLVM packs the struct the way clang likes, despite
    // our intents regarding the alignment of `inner` and
    // we end up with the same size `outer` as clang
    assert sys::size_of::<outer>() == 12u; // passes

    // But now our shape code doesn't find the inner struct
    // We print (22, (0))
    assert y == "(22, (44))"; // fails
}

#[cfg(target_arch = "x86_64")]
fn main() { }
