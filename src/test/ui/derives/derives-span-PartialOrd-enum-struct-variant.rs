// ignore-x86 FIXME: missing sysroot spans (#53081)
// This file was auto-generated using 'src/etc/generate-deriving-span-tests.py'

#[derive(PartialEq)]
struct Error;

#[derive(PartialOrd,PartialEq)]
enum Enum {
   A {
     x: Error //~ ERROR
   }
}

fn main() {}
