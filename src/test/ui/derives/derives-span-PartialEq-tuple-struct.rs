// This file was auto-generated using 'src/etc/generate-deriving-span-tests.py'


struct Error;

#[derive(PartialEq)]
struct Struct(
    Error //~ ERROR
//~^ ERROR
);

fn main() {}
