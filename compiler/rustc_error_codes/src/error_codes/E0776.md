`#[cmse_nonsecure_entry]` functions require a C ABI

Erroneous code example:

```compile_fail,E0776
#![feature(cmse_nonsecure_entry)]

#[no_mangle]
#[cmse_nonsecure_entry]
pub fn entry_function(input: Vec<u32>) {}
```

To fix this error, declare your entry function with a C ABI, using `extern "C"`.
