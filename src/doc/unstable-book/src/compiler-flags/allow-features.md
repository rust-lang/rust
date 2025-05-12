# `allow-features`

This feature is perma-unstable and has no tracking issue.

----

This flag allows limiting the features which can be enabled with `#![feature(...)]` attributes.
By default, all features are allowed on nightly and no features are allowed on stable or beta (but see [`RUSTC_BOOTSTRAP`]).

Features are comma-separated, for example `-Z allow-features=ffi_pure,f16`.
If the flag is present, any feature listed will be allowed and any feature not listed will be disallowed.
Any unrecognized feature is ignored.

[`RUSTC_BOOTSTRAP`]: ./rustc-bootstrap.html
