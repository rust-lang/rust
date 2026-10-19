## `tests/rustdoc-json/remap-path-prefix`

Tests for `--remap-path-prefix` and `--remap-path-scope` with rustdoc JSON:
 - `remap-path-prefix`: only uses bare `--remap-path-prefix`, expects remapped paths
 - `doc`: pass `--remap-path-prefix` and the `documentation` scope, expects remapped paths
 - `object`: pass `--remap-path-prefix` and the `object` scope, expects un-remapped paths
 - `debuginfo`: pass `--remap-path-prefix` and the `debuginfo` scope, expects un-remapped paths
