# `location-detail`

The tracking issue for this feature is: [#70580](https://github.com/rust-lang/rust/issues/70580).

------------------------

Option `-Z location-detail=val` controls what location details are tracked when
using `caller_location`. This allows users to control what location details
are printed as part of panic messages, by allowing them to exclude any combination
of filenames, line numbers, and column numbers. This option is intended to provide
users with a way to mitigate the size impact of `#[track_caller]`.

This option supports a comma separated list of location details to be included. Valid options
within this list are:

- `file` - the filename of the panic will be included in the panic output
- `line` - the source line of the panic will be included in the panic output
- `column` - the source column of the panic will be included in the panic output

Any combination of these three options are supported. Alternatively, you can pass
`none` to this option, which results in no location details being tracked.
If this option is not specified, all three are included by default.

An example of a panic output when using `-Z location-detail=line`:
```text
panicked at 'Process blink had a fault', <redacted>:323:0
```

The code size savings from this option are two-fold. First, the `&'static str` values
for each path to a file containing a panic are removed from the binary. For projects
with deep directory structures and many files with panics, this can add up. This category
of savings can only be realized by excluding filenames from the panic output. Second,
savings can be realized by allowing multiple panics to be fused into a single panicking
branch. It is often the case that within a single file, multiple panics with the same
panic message exist -- e.g. two calls to `Option::unwrap()` in a single line, or
two calls to `Result::expect()` on adjacent lines. If column and line information
are included in the `Location` struct passed to the panic handler, these branches cannot
be fused, as the output is different depending on which panic occurs. However if line
and column information is identical for all panics, these branches can be fused, which
can lead to substantial code size savings, especially for small embedded binaries with
many panics.

The savings from this option are amplified when combined with the use of `-Zbuild-std`, as
otherwise paths for panics within the standard library are still included in your binary.
