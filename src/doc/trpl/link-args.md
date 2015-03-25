% Link args

There is one other way to tell rustc how to customize linking, and that is via
the `link_args` attribute. This attribute is applied to `extern` blocks and
specifies raw flags which need to get passed to the linker when producing an
artifact. An example usage would be:

``` no_run
#![feature(link_args)]

#[link_args = "-foo -bar -baz"]
extern {}
# fn main() {}
```

Note that this feature is currently hidden behind the `feature(link_args)` gate
because this is not a sanctioned way of performing linking. Right now rustc
shells out to the system linker, so it makes sense to provide extra command line
arguments, but this will not always be the case. In the future rustc may use
LLVM directly to link native libraries in which case `link_args` will have no
meaning.

It is highly recommended to *not* use this attribute, and rather use the more
formal `#[link(...)]` attribute on `extern` blocks instead.

