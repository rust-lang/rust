# `env`

The tracking issue for this feature is: [#118372](https://github.com/rust-lang/rust/issues/118372).

------------------------

This option flag allows to specify environment variables value at compile time to be
used by `env!` and `option_env!` macros.

When retrieving an environment variable value, the one specified by `--env` will take
precedence. For example, if you want have `PATH=a` in your environment and pass:

```bash
rustc --env PATH=env
```

Then you will have:

```rust,no_run
assert_eq!(env!("PATH"), "env");
```

Please note that on Windows, environment variables are case insensitive but case
preserving whereas `rustc`'s environment variables are case sensitive. For example,
having `Path` in your environment (case insensitive) is different than using
`rustc --env Path=...` (case sensitive).
