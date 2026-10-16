 # `pointer-authentication`

The `-Zpointer-authentication` compiler flag controls pointer authentication options. It accepts a comma-separated list of options, each prefixed with `+` to enable the option or `-` to disable it.

This option is a target modifier and is currently intended for use with `aarch64-unknown-linux-pauthtest`.

## Syntax
```bash
rustc -Zpointer-authentication=+option,-option
```

Each option must be prefixed with `+` or `-`. Multiple options can be combined in a single, comma-separated, invocation. If an option appears more than once, the last occurrence wins.

Available options:

* `aarch64-jump-table-hardening` - enable hardened lowering for jump-table dispatch
* `auth-traps` - trap immediately on pointer authentication failure
* `calls` - enable signing and authentication of indirect calls
* `elf-got` - enable authentication of pointers from the Global Offset Table (ELF only)
* `function-pointer-type-discrimination` - enable type discrimination on function pointers
* `indirect-gotos` - enable signing and authentication of indirect goto targets
* `init-fini` - enable signing of function pointers in init/fini arrays
* `init-fini-address-discrimination` - enable address discrimination in init/fini arrays
* `intrinsics` - enable pointer authentication intrinsics
* `return-addresses` - enable signing and authentication of return addresses
* `typeinfo-vt-ptr-discrimination` - incorporate type and address discrimination in authenticated vtable pointers for `std::type_info`
* `vt-ptr-addr-discrimination` - incorporate address discrimination in authenticated vtable pointers
* `vt-ptr-type-discrimination` - incorporate type discrimination in authenticated vtable pointers

Please note, at present not all the options will have observable impact on the code generation - some are placeholders for future work, or provided for ABI compatibility (for example: `vt-ptr-addr-discrimination`).

## Examples


```bash
# Enable pointer authentication for indirect calls:
rustc -Zpointer-authentication=+calls

# Enable indirect-call authentication while disabling signing and authentication
# of function pointers in init/fini arrays:
rustc -Zpointer-authentication=+calls,-init-fini
```
