# Debugging

## How to debug GCC LTO

Run do the command with `-v -save-temps` and then extract the `lto1` line from the output and run that under the debugger.

## How to debug stdarch tests that cannot be ran locally

First, run the tests normally:

----
cd build/build_sysroot/sysroot_src/library/stdarch/
STDARCH_TEST_EVERYTHING=1 CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="sde -future -rtm_mode full --" TARGET=x86_64-unknown-linux-gnu ../../../../../y.sh cargo test
----

It will show the command it ran, something like this:

----
  process didn't exit successfully: `sde -future -rtm_mode full -- /home/user/projects/rustc_codegen_gcc/build/build_sysroot/sysroot_src/library/stdarch/target/debug/deps/core_arch-fd2d75f89baae5c6` (signal: 11, SIGSEGV: invalid memory reference)
----

Then add the `-debug` flag to it:

----
sde -debug -future -rtm_mode full -- /home/user/projects/rustc_codegen_gcc/build/build_sysroot/sysroot_src/library/stdarch/target/debug/deps/core_arch-fd2d75f89baae5c6
----

To see the symbols in `gdb`, specify the executable in your command:

----
gdb /home/user/projects/rustc_codegen_gcc/build/build_sysroot/sysroot_src/library/stdarch/target/debug/deps/core_arch-fd2d75f89baae5c6
----

and then write the `gdb` command that `sde` tells you to use, something like:

----
target remote :51299
----
