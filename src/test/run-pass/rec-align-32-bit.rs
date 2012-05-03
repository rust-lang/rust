// xfail-pretty
// xfail-win32
// Issue #2303

#[abi = "rust-intrinsic"]
native mod rusti {
    fn pref_align_of<T>() -> uint;
    fn min_align_of<T>() -> uint;
}

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

    #debug("align inner = %?", rusti::min_align_of::<inner>());
    #debug("size outer = %?", sys::size_of::<outer>());
    #debug("y = %s", y);

    // per clang/gcc the alignment of `inner` is 4 on x86.
    assert rusti::min_align_of::<inner>() == 4u;

    // per clang/gcc the size of `outer` should be 12
    // because `inner`s alignment was 4.
    assert sys::size_of::<outer>() == 12u;

    assert y == "(22, (44))";
}

#[cfg(target_arch = "x86_64")]
fn main() { }
