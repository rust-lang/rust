// ignore-x86
// ^ due to stderr output differences
// This file was auto-generated using 'src/etc/generate-deriving-span-tests.py'


struct Error;

#[derive(Hash)]
enum Enum {
   A {
     x: Error //~ ERROR
   }
}

fn main() {}
