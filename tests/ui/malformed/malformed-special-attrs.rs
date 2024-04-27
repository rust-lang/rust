#[cfg_attr] //~ ERROR malformed `cfg_attr` attribute
struct S1;

#[cfg_attr = ""] //~ ERROR malformed `cfg_attr` attribute
struct S2;

#[derive] //~ ERROR malformed `derive` attribute
struct S3;

#[derive = ""] //~ ERROR malformed `derive` attribute
struct S4;

fn main() {}
