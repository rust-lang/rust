# build-manifest

This tool generates the manifests uploaded to static.rust-lang.org and used by rustup.
You can see a full list of all manifests at <https://static.rust-lang.org/manifests.txt>.
This listing is updated by <https://github.com/rust-lang/generate-manifest-list> every 7 days.

This gets called by `promote-release` <https://github.com/rust-lang/promote-release>. `promote-release` downloads a pre-built binary of `build-manifest` which is generated in the dist-x86_64-linux builder and uploaded to s3.

## Adding a new component

1. Add a new `Step` to `dist.rs`. This should usually be named after the filename of the uploaded tarball. See https://github.com/rust-lang/rust/pull/101799/files#diff-2c56335faa24486df09ba392d8900c57e2fac4633e1f7038469bcf9ed3feb871 for an example.
    a. If appropriate, call `tarball.is_preview(true)` for the component.
2. Add a new `PkgType` to build-manifest. Fix all the compile errors as appropriate.

## Testing changes locally

In order to test the changes locally you need to have a valid dist directory
available locally. If you don't want to build all the compiler, you can easily
create one from the nightly artifacts with:

```sh
for component in rust rustc rust-std rust-docs cargo; do
    wget -P build/dist https://static.rust-lang.org/dist/${component}-nightly-x86_64-unknown-linux-gnu.tar.gz
done
```

Then, you can generate the manifest and all the packages from `build/dist` to
`build/manifest` with:

```sh
mkdir -p build/manifest
cargo +nightly run --release -p build-manifest build/dist build/manifest 1970-01-01 http://example.com nightly
```
