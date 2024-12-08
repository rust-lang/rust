// This file was auto-generated using 'src/etc/generate-deriving-span-tests.py'


struct Error;

#[derive(Default)]
struct Struct {
    x: Error //~ ERROR
}

fn main() {}
