# Custom Targets

If you'd like to build for a target that is not yet supported by `rustc`, you can use a
"custom target specification" to define a target. These target specification files
are JSON. To see the JSON for the host target, you can run:

```bash
rustc +nightly -Z unstable-options --print target-spec-json
```

To see it for a different target, add the `--target` flag:

```bash
rustc +nightly -Z unstable-options --target=wasm32-unknown-unknown --print target-spec-json
```

To use a custom target, see the (unstable) [`build-std` feature](../../cargo/reference/unstable.html#build-std) of `cargo`.

<div class="warning">

The target JSON properties are not stable and subject to change.
Always pin your compiler version when using custom targets!

</div>

## JSON Schema

`rustc` provides a JSON schema for the custom target JSON specification.
Because the schema is subject to change, you should always use the schema from the version of rustc which you are passing the target to.

It can be found in `etc/target-spec-json-schema.json` in the sysroot (`rustc --print sysroot`) or printed with `rustc +nightly -Zunstable-options --print target-spec-json-schema`.
The existence and name of this schema is, just like the properties of the JSON specification, not stable and subject to change.

## Custom Target Lookup Path

When `rustc` is given an option `--target=TARGET` (where `TARGET` is any string), it uses the following logic:
1. if `TARGET` is the name of a built-in target, use that
2. if `TARGET` is a path to a file, read that file as a json target
3. otherwise, search the colon-separated list of directories found
   in the `RUST_TARGET_PATH` environment variable from left to right
   for a file named `TARGET.json`.

These steps are tried in order, so if there are multiple potentially valid
interpretations for a target, whichever is found first will take priority.
If none of these methods find a target, an error is thrown.
