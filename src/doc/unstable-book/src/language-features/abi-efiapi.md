# `abi_efiapi`

The tracking issue for this feature is: [#65815]

[#65815]: https://github.com/rust-lang/rust/issues/65815

------------------------

The `efiapi` calling convention can be used for defining a function with
an ABI compatible with the UEFI Interfaces as defined in the [UEFI
Specification].

Example:

```rust
#![feature(abi_efiapi)]

extern "efiapi" { fn f1(); }

extern "efiapi" fn f2() { todo!() }
```

[UEFI Specification]: https://uefi.org/specs/UEFI/2.10/
