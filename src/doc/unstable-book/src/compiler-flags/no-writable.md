# `no-writable`

---

This flag will globally stop the compiler from inserting the [writable](https://llvm.org/docs/LangRef.html#writable) LLVM flag. It also stops [Miri](https://github.com/rust-lang/miri) from testing for undefined behavior when inserting writes. It has the same effect as applying `#[rustc_no_writable]` to every function.
