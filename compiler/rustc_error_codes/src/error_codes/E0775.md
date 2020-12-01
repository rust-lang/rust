`#[cmse_nonsecure_entry]` is only valid for targets with the TrustZone-M
extension.

Erroneous code example:

```compile_fail,E0775
#![feature(cmse_nonsecure_entry)]

#[cmse_nonsecure_entry]
pub extern "C" fn entry_function() {}
```

To fix this error, compile your code for a Rust target that supports the
TrustZone-M extension. The current possible targets are:
* `thumbv8m.main-none-eabi`
* `thumbv8m.main-none-eabihf`
* `thumbv8m.base-none-eabi`
