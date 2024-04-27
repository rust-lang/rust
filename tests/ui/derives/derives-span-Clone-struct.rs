// This file was auto-generated using 'src/etc/generate-deriving-span-tests.py'


struct Error;

#[derive(Clone)]
struct Struct {
    x: Error //~ ERROR
}

fn main() {}
