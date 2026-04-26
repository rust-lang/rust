# `llvm-writable`

---

Setting this flag will allow the compiler to insert the [writable](https://llvm.org/docs/LangRef.html#writable) LLVM flag.
This allows for more optimizations but also introduces more Undefined Behaviour.
To be more precise, mutable reference function arguments are now considered to be always writable, which means the compiler may insert writes to those references even if the original code contained no such writes.
The attribute `#[rustc_no_writable]` can be used to disable the optimization on a per-function basis.

The [Miri](https://github.com/rust-lang/miri) tool can be used to detect some problematic cases.
However, note that when using Tree Borrows, you must set `-Zmiri-tree-borrows-implicit-writes` to ensure that the UB arising from these implicit writes is detected.
