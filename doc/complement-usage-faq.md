% Usage FAQ

# How do I get my program to display the output of `log` statements?

**Short answer** set the RUST_LOG environment variable to the name of your source file, sans extension.

```sh
rustc hello.rs
export RUST_LOG=hello
./hello
```

**Long answer** RUST_LOG takes a 'logging spec' that consists of a comma-separated list of paths, where a path consists of the crate name and sequence of module names, each separated by double-colons. For standalone .rs files the crate is implicitly named after the source file, so in the above example we were setting RUST_LOG to the name of the hello crate. Multiple paths can be combined to control the exact logging you want to see. For example, when debugging linking in the compiler you might set `RUST_LOG=rustc::metadata::creader,rustc::util::filesearch,rustc::back::rpath`

If you aren't sure which paths you need, try setting RUST_LOG to `::help` and running your program. This will print a list of paths available for logging. For a full description see [the language reference][1].

[1]:http://doc.rust-lang.org/doc/master/rust.html#logging-system

# How do I get my program to display the output of `debug!` statements?

This is much like the answer for `log` statements, except that you also need to compile your program in debug mode (that is, pass `--cfg debug` to `rustc`).  Note that if you want to see the instrumentation of the `debug!` statements within `rustc` itself, you need a debug version of `rustc`; you can get one by invoking `configure` with the `--enable-debug` option.

# What does it mean when a program exits with `leaked memory in rust main loop (2 objects)' failed, rt/memory_region.cpp:99 2 objects`?

This message indicates a memory leak, and is mostly likely to happen on rarely exercised failure paths. Note that failure unwinding is not yet implemented on windows so this is expected. If you see this on Linux or Mac it's a compiler bug; please report it.

# Why do gdb backtraces end with the error 'previous frame inner to this frame (corrupt stack?)'?

**Short answer** your gdb is too old to understand our hip new stacks. Upgrade to a newer version (7.3.1 is known to work).

**Long answer** Rust uses 'spaghetti stacks' (a linked list of stacks) to allow tasks to start very small but recurse arbitrarily deep when necessary. As a result, new frames don't always decrease the stack pointer like gdb expects but instead may jump around the heap to different stack segments. Newer versions of gdb recognize that the special function called __morestack may change the stack pointer to a different stack.

# Why did my build create a bunch of zero-length files in my lib directory?

This is a normal part of the Rust build process. The build system uses these zero-length files for dependency tracking, as the actual names of the Rust libraries contain hashes that can't be guessed easily by the Makefiles.
