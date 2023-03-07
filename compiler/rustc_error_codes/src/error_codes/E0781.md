The `C-cmse-nonsecure-call` ABI can only be used with function pointers.

Erroneous code example:

```compile_fail,E0781
#![feature(abi_c_cmse_nonsecure_call)]

pub extern "C-cmse-nonsecure-call" fn test() {}
```

The `C-cmse-nonsecure-call` ABI should be used by casting function pointers to
specific addresses.
