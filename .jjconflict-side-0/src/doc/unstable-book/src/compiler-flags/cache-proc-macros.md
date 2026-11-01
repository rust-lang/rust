## `cache-proc-macros`

The tracking issue for this feature is: [#151364]

[#151364]: https://github.com/rust-lang/rust/issues/151364

------------------------

This option instructs `rustc` to cache (derive) proc-macro invocations using the incremental system. Note that the compiler does not currently check whether the proc-macro is actually "cacheable" or not. If you use this flag when compiling a crate that uses non-pure proc-macros, it can result in stale expansions being compiled.
