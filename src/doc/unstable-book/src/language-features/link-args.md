# `link_args`

The tracking issue for this feature is: [#29596]

[#29596]: https://github.com/rust-lang/rust/issues/29596

------------------------

You can tell `rustc` how to customize linking, and that is via the `link_args`
attribute. This attribute is applied to `extern` blocks and specifies raw flags
which need to get passed to the linker when producing an artifact. An example
usage would be:

```rust,no_run
#![feature(link_args)]

#[link_args = "-foo -bar -baz"]
extern "C" {}
# fn main() {}
```

Note that this feature is currently hidden behind the `feature(link_args)` gate
because this is not a sanctioned way of performing linking. Right now `rustc`
shells out to the system linker (`gcc` on most systems, `link.exe` on MSVC), so
it makes sense to provide extra command line arguments, but this will not
always be the case. In the future `rustc` may use LLVM directly to link native
libraries, in which case `link_args` will have no meaning. You can achieve the
same effect as the `link_args` attribute with the `-C link-args` argument to
`rustc`.

It is highly recommended to *not* use this attribute, and rather use the more
formal `#[link(...)]` attribute on `extern` blocks instead.
