# Security

At the moment, rust-analyzer assumes that all code is trusted. Here is a
**non-exhaustive** list of ways to make rust-analyzer execute arbitrary
code:

-   proc macros and build scripts are executed by default

-   `.cargo/config` can override `rustc` with an arbitrary executable

-   `rust-toolchain.toml` can override `rustc` with an arbitrary
    executable

-   VS Code plugin reads configuration from project directory, and that
    can be used to override paths to various executables, like `rustfmt`
    or `rust-analyzer` itself.

-   rust-analyzer’s syntax trees library uses a lot of `unsafe` and
    hasn’t been properly audited for memory safety.