#[cfg_attr] //~ ERROR bad `cfg_attr` attribute
struct S1;

#[cfg_attr = ""] //~ ERROR expected `(`, found `=`
struct S2;

#[derive] //~ ERROR bad `derive` attribute
struct S3;

#[derive = ""] //~ ERROR bad `derive` attribute
struct S4;

fn main() {}
