#![feature(compiler_builtins, staged_api)]
#![compiler_builtins]
#![no_std]

#![unstable(
    feature = "compiler_builtins_lib",
    reason = "Compiler builtins. Will never become stable.",
    issue = "0"
)]
