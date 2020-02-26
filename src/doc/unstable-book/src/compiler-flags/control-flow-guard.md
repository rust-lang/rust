# `control_flow_guard`

The tracking issue for this feature is: [#68793](https://github.com/rust-lang/rust/issues/68793).

------------------------

The `-Zcontrol_flow_guard=checks` compiler flag enables the Windows [Control Flow Guard][cfguard-docs] platform security feature. When enabled, the compiler outputs a list of valid indirect call targets, and inserts runtime checks on all indirect jump instructions to ensure that the destination is in the list of valid call targets.

[cfguard-docs]: https://docs.microsoft.com/en-us/windows/win32/secbp/control-flow-guard

For testing purposes, the `-Zcontrol_flow_guard=nochecks` compiler flag can be used to emit only the list of valid call targets, but not the runtime checks.

It is strongly recommended to also enable Control Flow Guard checks in all linked libraries, including the standard library. 

To enable Control Flow Guard in the standard library, you can use the [cargo `-Zbuild-std` functionality][build-std] to recompile the standard library with the same configuration options as the main program. 

[build-std]: https://doc.rust-lang.org/nightly/cargo/reference/unstable.html#build-std

For example:
```cmd
rustup toolchain install --force nightly
rustup component add rust-src
SET RUSTFLAGS=-Zcontrol_flow_guard=checks
cargo +nightly build -Z build-std --target x86_64-pc-windows-msvc
```

```PowerShell
rustup toolchain install --force nightly
rustup component add rust-src
$Env:RUSTFLAGS = "-Zcontrol_flow_guard=checks"
cargo +nightly build -Z build-std --target x86_64-pc-windows-msvc
```

Alternatively, if you are building the standard library from source, you can set `control-flow-guard = true` in the config.toml file.
