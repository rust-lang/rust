// Unknown `{name}` placeholders in `--remap-path-prefix` are rejected.
// A literal `{foo}` directory would be written `{{foo}}`.
//
//@ compile-flags: --remap-path-prefix={foo}=bar

fn main() {}

//~? ERROR unknown placeholder `{foo}` in `--remap-path-prefix`
