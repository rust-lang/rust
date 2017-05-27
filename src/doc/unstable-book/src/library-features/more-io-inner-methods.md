# `more_io_inner_methods`

The tracking issue for this feature is: [#41519]

[#41519]: https://github.com/rust-lang/rust/issues/41519

------------------------

This feature enables several internal accessor methods on structures in
`std::io` including `Take::{get_ref, get_mut}` and `Chain::{into_inner, get_ref,
get_mut}`.
