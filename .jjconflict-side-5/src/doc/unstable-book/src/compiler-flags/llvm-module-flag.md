# `llvm-module-flag`

---------------------

This flag allows adding a key/value to the `!llvm.module.flags` metadata in the
LLVM-IR for a compiled Rust module.  The syntax is

`-Z llvm_module_flag=<name>:<type>:<value>:<behavior>`

Currently only u32 values are supported but the type is required to be specified
for forward compatibility.  The `behavior` element must match one of the named
LLVM [metadata behaviors](https://llvm.org/docs/LangRef.html#module-flags-metadata)
