# `llvm-writable`

---

Setting this flag will allow the compiler to insert the [writable](https://llvm.org/docs/LangRef.html#writable) LLVM flag. This allows for more optimizations but also introduces more Undefined Behaviour. To be more precise, mutable borrows on function entry are now considered to be always writable and there should be no new Undefined Behaviour when the compiler tries to write to them even if there was no write in the original source code. The [Miri](https://github.com/rust-lang/miri) tool can be used to detect some problematic cases. The attribute `#[rustc_no_writable]` can be used to disable the optimization on a per function basis.
