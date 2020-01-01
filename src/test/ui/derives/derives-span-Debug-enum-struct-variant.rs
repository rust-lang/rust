// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl
// This file was auto-generated using 'src/etc/generate-deriving-span-tests.py'


struct Error;

#[derive(Debug)]
enum Enum {
   A {
     x: Error //~ ERROR
   }
}

fn main() {}
