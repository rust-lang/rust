# build-manifest

This tool generates the manifests uploaded to static.rust-lang.org and used by rustup.
You can see a full list of all manifests at <https://static.rust-lang.org/manifests.txt>.

We auto-generate the host targets (those with full compiler toolchains) and
target targets (a superset of hosts, some of which only support std) through
`build.rs`, which internally uses a stage 1 rustc to produce the target list
and uses the `TargetMetadata` to determine whether host tools are expected and
whether artifacts are expected. This list is not currently verified against the
actually produced artifacts by CI, though that may change in the future.

This gets called by `promote-release` <https://github.com/rust-lang/promote-release>. `promote-release` downloads a pre-built binary of `build-manifest` which is generated in the dist-x86_64-linux builder and uploaded to s3.

## Adding a new component

1. Add a new `Step` to `dist.rs`. This should usually be named after the filename of the uploaded tarball. See https://github.com/rust-lang/rust/pull/101799/files#diff-2c56335faa24486df09ba392d8900c57e2fac4633e1f7038469bcf9ed3feb871 for an example.
    a. If appropriate, call `tarball.is_preview(true)` for the component.
2. Add a new `PkgType` to build-manifest. Fix all the compile errors as appropriate.

## Testing changes locally with rustup

In order to test the changes locally you need to have a valid dist directory
available locally. If you don't want to build all the compiler, you can easily
create one from the nightly artifacts with:

```sh
for component in rust rustc rust-std rust-docs cargo; do
    wget -P build/dist https://static.rust-lang.org/dist/${component}-nightly-x86_64-unknown-linux-gnu.tar.xz
done
```

Then, you can generate the manifest and add it to `build/dist`:

```sh
cargo +nightly run --release -p build-manifest build/dist build/dist 1970-01-01 http://localhost:8000 nightly
```

After that, generate a SHA256 stamp for the manifest file:
```sh
sha256sum build/dist/channel-rust-nightly.toml > build/dist/channel-rust-nightly.toml.sha256
```

And start a HTTP server from the `build` directory:
```sh
cd build
python3 -m http.server 8000
```

After you do all that, you can then install the locally generated components with rustup:
```
rustup uninstall nightly
RUSTUP_DIST_SERVER=http://localhost:8000 rustup toolchain install nightly --profile minimal
RUSTUP_DIST_SERVER=http://localhost:8000 rustup +nightly component add <my-new-component>
```

Note that generally it will not work to combine components built locally and those built from CI (nightly). Ideally, if you want to ship new rustup components, first dist them in nightly, and then test everything from nightly here after it's available on CI.
