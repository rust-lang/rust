# `fat-lto-crates`

------------------------

With `-Clto=thin`, `-Zfat-lto-crates=crate_a,crate_b` merges all codegen
units of the named crates into a single module before the ThinLTO link, the
way `-Clto=fat` merges the whole program. The merged "core" then takes part
in the ThinLTO link as an ordinary module: the combined summary index still
performs global dead-symbol elimination, internalization, and import
analysis across the whole program, and functions are imported across the
core's boundary under the usual ThinLTO heuristics. The core itself is
optimized with the fat LTO pipeline, so the named crates receive
unrestricted whole-program optimization among themselves.

Selected crates receive fat-LTO optimization while the remaining crates stay
as separate ThinLTO modules.

The flag combines with `-Zfat-lto-partitions`, which then splits the merged
core for parallel codegen.

Crate names may be written with `-` or `_`. Names that match no crate in
the link are ignored. The flag only has an effect with the LLVM backend and
`-Clto=thin`. Crates not named by the flag stay outside the core.

With incremental compilation, rustc caches the merged pre-LTO core and its
post-LTO objects. Tracked compiler options first gate all prior work products.
Member names and LLVM's full-module hashes then key the merged bitcode. Rustc
checksums it before passing it to LLVM. LLVM's ThinLTO cache key independently
guards the objects and includes cross-module inputs such as imports and exports.
Changing a core body rebuilds the merge. Changing an imported function outside
the core can reuse the merge but rebuilds its objects. A partitioned core is
reused only when every partition object is available.

Example, with `lto = "thin"` in the Cargo profile:

```console
RUSTFLAGS="-Zfat-lto-crates=my_app,my_engine" cargo +nightly build --release
```
