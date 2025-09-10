# GCC codegen backend tests

To test the GCC codegen backend, you need to add `"gcc"` into the `rust.codegen-backends`
setting in `bootstrap.toml`:

```toml
rust.codegen-backends = ["llvm", "gcc"]
```

If you don't want to change your `bootstrap.toml` file, you can alternatively run your `x.py`
commands with `--set rust.codegen-backends=["llvm", "gcc"]'`. For example:

```bash
x.py test --set 'rust.codegen-backends=["llvm", "gcc"]'
```

If you don't want to build `gcc` yourself, you also need to set:

```toml
gcc.download-ci-gcc = true
```

Then when running tests, add the `--test-codegen-backend gcc` option. For example:

```bash
./x.py test tests/ui --test-codegen-backend gcc
```

If you want to build the sysroot using the GCC backend, you need to set it first
in `rust.codegen-backends`:

```toml
rust.codegen-backends = ["llvm", "gcc"]
```
