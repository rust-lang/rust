# `linker-flavor`

The tracking issue for this feature is: None

------------------------

Every `rustc` target defaults to some linker. For example, Linux targets default
to gcc. In some cases, you may want to override the default; you can do that
with the unstable CLI argument: `-Z linker-flavor`.

Here how you would use this flag to link a Rust binary for the
`thumbv7m-none-eabi` using LLD instead of GCC.

``` text
$ xargo rustc --target thumbv7m-none-eabi -- \
    -C linker=ld.lld \
    -Z linker-flavor=ld \
    -Z print-link-args | tr ' ' '\n'
"ld.lld"
"-L"
"$SYSROOT/lib/rustlib/thumbv7m-none-eabi/lib"
"$PWD/target/thumbv7m-none-eabi/debug/deps/app-512e9dbf385f233c.0.o"
"-o"
"$PWD/target/thumbv7m-none-eabi/debug/deps/app-512e9dbf385f233c"
"--gc-sections"
"-L"
"$PWD/target/thumbv7m-none-eabi/debug/deps"
"-L"
"$PWD/target/debug/deps"
"-L"
"$SYSROOT/lib/rustlib/thumbv7m-none-eabi/lib"
"-Bstatic"
"$SYSROOT/lib/rustlib/thumbv7m-none-eabi/lib/libcore-e1ccb7dfb1cb9ebb.rlib"
"-Bdynamic"
```

Whereas the default is:

``` text
$ xargo rustc --target thumbv7m-none-eabi -- \
    -C link-arg=-nostartfiles \
    -Z print-link-args | tr ' ' '\n'
"arm-none-eabi-gcc"
"-L"
"$SYSROOT/lib/rustlib/thumbv7m-none-eabi/lib"
"$PWD/target/thumbv7m-none-eabi/debug/deps/app-961e39416baa38d9.0.o"
"-o"
"$PWD/target/thumbv7m-none-eabi/debug/deps/app-961e39416baa38d9"
"-Wl,--gc-sections"
"-nodefaultlibs"
"-L"
"$PWD/target/thumbv7m-none-eabi/debug/deps"
"-L"
"$PWD/target/debug/deps"
"-L"
"$SYSROOT/lib/rustlib/thumbv7m-none-eabi/lib"
"-Wl,-Bstatic"
"$SYSROOT/lib/rustlib/thumbv7m-none-eabi/lib/libcore-e1ccb7dfb1cb9ebb.rlib"
"-nostartfiles"
"-Wl,-Bdynamic"
```
