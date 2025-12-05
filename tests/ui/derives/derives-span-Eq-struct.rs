// This file was auto-generated using 'src/etc/generate-deriving-span-tests.py'

#[derive(PartialEq)]
struct Error;

#[derive(Eq,PartialEq)]
struct Struct {
    x: Error //~ ERROR
}

fn main() {}
