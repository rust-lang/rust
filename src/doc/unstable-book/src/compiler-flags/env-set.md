# `env-set`

The tracking issue for this feature is: [#118372](https://github.com/rust-lang/rust/issues/118372).

------------------------

This option flag allows to specify environment variables value at compile time to be
used by `env!` and `option_env!` macros. It also impacts `tracked_env::var` function
from the `proc_macro` crate.

This information will be stored in the dep-info files. For more information about
dep-info files, take a look [here](https://doc.rust-lang.org/cargo/guide/build-cache.html#dep-info-files).

When retrieving an environment variable value, the one specified by `--env-set` will take
precedence. For example, if you want have `PATH=a` in your environment and pass:

```bash
rustc --env-set PATH=env
```

Then you will have:

```rust,no_run
assert_eq!(env!("PATH"), "env");
```

It will trigger a new compilation if any of the `--env-set` argument value is different.
So if you first passed:

```bash
--env-set A=B --env X=12
```

and then on next compilation:

```bash
--env-set A=B
```

`X` value is different (not set) so the code will be re-compiled.

Please note that on Windows, environment variables are case insensitive but case
preserving whereas `rustc`'s environment variables are case sensitive. For example,
having `Path` in your environment (case insensitive) is different than using
`rustc --env-set Path=...` (case sensitive).
