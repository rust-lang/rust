# build-manifest

This tool generates the manifests uploaded to static.rust-lang.org and used by
rustup. The tool is invoked by the bootstrap tool.

## Testing changes locally

In order to test the changes locally you need to have a valid dist directory
available locally. If you don't want to build all the compiler, you can easily
create one from the nightly artifacts with:

```
#!/bin/bash
for cmpn in rust rustc rust-std rust-docs cargo; do
    wget https://static.rust-lang.org/dist/${cmpn}-nightly-x86_64-unknown-linux-gnu.tar.gz
done
```

Then, you can generate the manifest and all the packages from `path/to/dist` to
`path/to/output` with:

```
$ cargo +nightly run path/to/dist path/to/output 1970-01-01 http://example.com CHANNEL
```

Remember to replace `CHANNEL` with the channel you produced dist artifacts of
and `VERSION` with the current Rust version.
