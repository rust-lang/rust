- Feature Name: dllimport
- Start Date: 2016-08-13
- RFC PR: [rust-lang/rfcs#1717](https://github.com/rust-lang/rfcs/pull/1717)
- Rust Issue: [rust-lang/rust#37403](https://github.com/rust-lang/rust/issues/37403)

# Summary
[summary]: #summary

Make compiler aware of the association between library names adorning `extern` blocks
and symbols defined within the block.  Add attributes and command line switches that leverage
this association.

# Motivation
[motivation]: #motivation

Most of the time a linkage directive is only needed to inform the linker about
what native libraries need to be linked into a program. On some platforms,
however, the compiler needs more detailed knowledge about what's being linked
from where in order to ensure that symbols are wired up correctly.

On Windows, when a symbol is imported from a dynamic library, the code that accesses
this symbol must be generated differently than for symbols imported from a static library.

Currently the compiler is not aware of associations between the libraries and symbols
imported from them, so it cannot alter code generation based on library kind.

# Detailed design
[design]: #detailed-design

### Library <-> symbol association

The compiler shall assume that symbols defined within extern block
are imported from the library mentioned in the `#[link]` attribute adorning the block.

### Changes to code generation

On platforms other than Windows the above association will have no effect.
On Windows, however, `#[link(..., kind="dylib")` shall be presumed to mean linking to a dll,
whereas `#[link(..., kind="static")` shall mean static linking.  In the former case, all symbols
associated with that library will be marked with LLVM [dllimport][1] storage class.

[1]: http://llvm.org/docs/LangRef.html#dll-storage-classes

### Library name and kind variance

Many native libraries are linked via the command line via `-l` which is passed
in through Cargo build scripts instead of being written in the source code
itself. As a recap, a native library may change names across platforms or
distributions or it may be linked dynamically in some situations and
statically in others which is why build scripts are leveraged to make these
dynamic decisions. In order to support this kind of dynamism, the following
modifications are proposed:

- Extend syntax of the `-l` flag to `-l [KIND=]lib[:NEWNAME]`.  The `NEWNAME`
  part may be used to override name of a library specified in the source.
- Add new meaning to the `KIND` part: if "lib" is already specified in the source,
  this will override its kind with KIND.  Note that this override is possible only
  for libraries defined in the current crate.

Example:

```rust
// mylib.rs
#[link(name="foo", kind="dylib")]
extern {
    // dllimport applied
}

#[link(name="bar", kind="static")]
extern {
    // dllimport not applied
}

#[link(name="baz")]
extern {
    // kind defaults to "dylib", dllimport applied
}
```

```sh
rustc mylib.rs -l static=foo # change foo's kind to "static", dllimport will not be applied
rustc mylib.rs -l foo:newfoo # link newfoo instead of foo, keeping foo's kind as "dylib"
rustc mylib.rs -l dylib=bar # change bar's kind to "dylib", dllimport will be applied
```

### Unbundled static libs (optional)

It had been pointed out that sometimes one may wish to link to a static system library
(i.e. one that is always available to the linker) without bundling it into .lib's and .rlib's.
For this use case we'll introduce another library "kind", "static-nobundle".
Such libraries would be treated in the same way as "static", except they will not be bundled into
the target .lib/.rlib.

# Drawbacks
[drawbacks]: #drawbacks

For libraries to work robustly on MSVC, the correct `#[link]` annotation will
be required. Most cases will "just work" on MSVC due to the compiler strongly
favoring static linkage, but any symbols imported from a dynamic library or
exported as a Rust dynamic library will need to be tagged appropriately to
ensure that they work in all situations. Worse still, the `#[link]` annotations
on an `extern` block are not required on any other platform to work correctly,
meaning that it will be common that these attributes are left off by accident.


# Alternatives
[alternatives]: #alternatives

- Instead of enhancing `#[link]`, a `#[linked_from = "foo"]` annotation could be added.
  This has the drawback of not being able to handle native libraries whose
  name is unpredictable across platforms in an easy fashion, however.
  Additionally, it adds an extra attribute to the comipler that wasn't known
  previously.

- Support a `#[dllimport]` on extern blocks (or individual symbols, or both).
  This has the following drawbacks, however:
  - This attribute would duplicate the information already provided by
    `#[link(kind="...")]`.
  - It is not always known whether `#[dllimport]` is needed. Native
    libraires are not always known whether they're linked dynamically or
    statically (e.g. that's what a build script decides), so `dllimport`
    will need to be guarded by `cfg_attr`.

- When linking native libraries, the compiler could attempt to locate each
  library on the filesystem and probe the contents for what symbol names are
  exported from the native library. This list could then be cross-referenced
  with all symbols declared in the program locally to understand which symbols
  are coming from a dylib and which are being linked statically. Some downsides
  of this approach may include:

    - It's unclear whether this will be a performant operation and not cause
      undue runtime overhead during compiles.

    - On Windows linking to a DLL involves linking to its "import library", so
      it may be difficult to know whether a symbol truly comes from a DLL or
      not.

    - Locating libraries on the system may be difficult as the system linker
      often has search paths baked in that the compiler does not know about.

- As was already mentioned, "kind" override can affect codegen of the current crate only.
  This overloading the `-l` flag for this purpose may be confusinfg to developers.
  A new codegen flag might be a better fit for this, for example `-C libkind=KIND=LIB`.

# Unresolved questions
[unresolved]: #unresolved-questions

- Should we allow dropping a library specified in the source from linking via `-l lib:` (i.e. "rename to empty")?
