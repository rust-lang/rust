// xfail-pretty
// Issue #2303

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn pref_align_of<T>() -> uint;
    fn min_align_of<T>() -> uint;
}

// This is the type with the questionable alignment
type inner = {
    c64: u32
};

// This is the type that contains the type with the
// questionable alignment, for testing
type outer = {
    c8: u8,
    t: inner
};


#[cfg(target_arch = "x86")]
mod m {
    fn align() -> uint { 4u }
    fn size() -> uint { 8u }
}

#[cfg(target_arch = "x86_64")]
mod m {
    fn align() -> uint { 4u }
    fn size() -> uint { 8u }
}

fn main() {

    let x = {c8: 22u8, t: {c64: 44u32}};

    // Send it through the shape code
    let y = #fmt["%?", x];

    #debug("align inner = %?", rusti::min_align_of::<inner>());
    #debug("size outer = %?", sys::size_of::<outer>());
    #debug("y = %s", y);

    // per clang/gcc the alignment of `inner` is 4 on x86.
    assert rusti::min_align_of::<inner>() == m::align();

    // per clang/gcc the size of `outer` should be 12
    // because `inner`s alignment was 4.
    assert sys::size_of::<outer>() == m::size();

    assert y == "(22, (44))";
}
