// This file was auto-generated using 'src/etc/generate-deriving-span-tests.py'


struct Error;

#[derive(Clone)]
struct Struct(
    Error //~ ERROR
);

fn main() {}
