# `--print` Options

The behavior of the `--print` flag can be modified by optionally be specifiying a filepath
for each requested information kind, in the format `--print KIND=PATH`, just like for
`--emit`. When a path is specified, information will be written there instead of to stdout.

This is unstable feature, so you have to provide `-Zunstable-options` to enable it.

## Examples

`rustc main.rs -Z unstable-options --print cfg=cfgs.txt`
