// See issue #84108 -- this is a test to ensure we do not ICE
// on this invalid code.

#![crate_type = "lib"]

static FOO: (dyn AsRef<OsStr>, u8) = ("hello", 42);
//~^ ERROR cannot find type `OsStr` in this scope

const BAR: (&Path, [u8], usize) = ("hello", [], 42);
//~^ ERROR cannot find type `Path` in this scope
//~| ERROR the size for values of type `[u8]` cannot be known at compilation time
//~| ERROR the size for values of type `[u8]` cannot be known at compilation time
//~| ERROR mismatched types

static BAZ: ([u8], usize) = ([], 0);
//~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
