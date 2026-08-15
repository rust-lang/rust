// Compiler:

// Regression test for <https://github.com/rust-lang/rustc_codegen_gcc/issues/836>

#![crate_type = "lib"]

#[unsafe(no_mangle)]
extern "C" fn third(_a: usize, b: usize, c: usize) {
    let throw_away_f: fn((), usize, usize) =
        unsafe { std::mem::transmute(third as extern "C" fn(_, _, _)) };
    throw_away_f((), 2, 3)
}
