# `remap-path-prefix`

The tracking issue for this feature is: [#41555](https://github.com/rust-lang/rust/issues/41555)

------------------------

The `-Z remap-path-prefix-from`, `-Z remap-path-prefix-to` commandline option
pair allows to replace prefixes of any file paths the compiler emits in various
places. This is useful for bringing debuginfo paths into a well-known form and
for achieving reproducible builds independent of the directory the compiler was
executed in. All paths emitted by the compiler are affected, including those in
error messages.

In order to map all paths starting with `/home/foo/my-project/src` to
`/sources/my-project`, one would invoke the compiler as follows:

```text
rustc -Zremap-path-prefix-from="/home/foo/my-project/src" -Zremap-path-prefix-to="/sources/my-project"
```

Debuginfo for code from the file `/home/foo/my-project/src/foo/mod.rs`,
for example, would then point debuggers to `/sources/my-project/foo/mod.rs`
instead of the original file.

The options can be specified multiple times when multiple prefixes should be
mapped:

```text
rustc -Zremap-path-prefix-from="/home/foo/my-project/src" \
      -Zremap-path-prefix-to="/sources/my-project" \
      -Zremap-path-prefix-from="/home/foo/my-project/build-dir" \
      -Zremap-path-prefix-to="/stable-build-dir"
```

When the options are given multiple times, the nth `-from` will be matched up
with the nth `-to` and they can appear anywhere on the commandline. Mappings
specified later on the line will take precedence over earlier ones.
