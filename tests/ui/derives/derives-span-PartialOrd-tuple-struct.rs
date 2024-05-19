// This file was auto-generated using 'src/etc/generate-deriving-span-tests.py'

#[derive(PartialEq)]
struct Error;

#[derive(PartialOrd,PartialEq)]
struct Struct(
    Error //~ ERROR can't compare `Error` with `Error`
);

fn main() {}
