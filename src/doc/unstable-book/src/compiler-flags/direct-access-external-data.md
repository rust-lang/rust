# `direct_access_external_data`

The tracking issue for this feature is: https://github.com/rust-lang/compiler-team/issues/707

------------------------

Option `-Z direct-access-external-data` controls how to access symbols of
external data.

Supported values for this option are:

- `yes` - Don't use GOT indirection to reference external data symbols.
- `no` - Use GOT indirection to reference external data symbols.

If the option is not explicitly specified, different targets have different
default values.
