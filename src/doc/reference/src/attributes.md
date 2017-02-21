# Attributes

Any item declaration may have an _attribute_ applied to it. Attributes in Rust
are modeled on Attributes in ECMA-335, with the syntax coming from ECMA-334
(C#). An attribute is a general, free-form metadatum that is interpreted
according to name, convention, and language and compiler version. Attributes
may appear as any of:

* A single identifier, the attribute name
* An identifier followed by the equals sign '=' and a literal, providing a
  key/value pair
* An identifier followed by a parenthesized list of sub-attribute arguments

Attributes with a bang ("!") after the hash ("#") apply to the item that the
attribute is declared within. Attributes that do not have a bang after the hash
apply to the item that follows the attribute.

An example of attributes:

```{.rust}
// General metadata applied to the enclosing module or crate.
#![crate_type = "lib"]

// A function marked as a unit test
#[test]
fn test_foo() {
    /* ... */
}

// A conditionally-compiled module
#[cfg(target_os="linux")]
mod bar {
    /* ... */
}

// A lint attribute used to suppress a warning/error
#[allow(non_camel_case_types)]
type int8_t = i8;
```

> **Note:** At some point in the future, the compiler will distinguish between
> language-reserved and user-available attributes. Until then, there is
> effectively no difference between an attribute handled by a loadable syntax
> extension and the compiler.

## Crate-only attributes

- `crate_name` - specify the crate's crate name.
- `crate_type` - see [linkage](linkage.html).
- `feature` - see [compiler features](#compiler-features).
- `no_builtins` - disable optimizing certain code patterns to invocations of
                  library functions that are assumed to exist
- `no_main` - disable emitting the `main` symbol. Useful when some other
   object being linked to defines `main`.
- `no_start` - disable linking to the `native` crate, which specifies the
  "start" language item.
- `no_std` - disable linking to the `std` crate.
- `plugin` - load a list of named crates as compiler plugins, e.g.
             `#![plugin(foo, bar)]`. Optional arguments for each plugin,
             i.e. `#![plugin(foo(... args ...))]`, are provided to the plugin's
             registrar function.  The `plugin` feature gate is required to use
             this attribute.
- `recursion_limit` - Sets the maximum depth for potentially
                      infinitely-recursive compile-time operations like
                      auto-dereference or macro expansion. The default is
                      `#![recursion_limit="64"]`.

### Module-only attributes

- `no_implicit_prelude` - disable injecting `use std::prelude::*` in this
  module.
- `path` - specifies the file to load the module from. `#[path="foo.rs"] mod
  bar;` is equivalent to `mod bar { /* contents of foo.rs */ }`. The path is
  taken relative to the directory that the current module is in.

## Function-only attributes

- `main` - indicates that this function should be passed to the entry point,
  rather than the function in the crate root named `main`.
- `plugin_registrar` - mark this function as the registration point for
  [compiler plugins][plugin], such as loadable syntax extensions.
- `start` - indicates that this function should be used as the entry point,
  overriding the "start" language item. See the "start" [language
  item](#language-items) for more details.
- `test` - indicates that this function is a test function, to only be compiled
  in case of `--test`.
- `should_panic` - indicates that this test function should panic, inverting the success condition.
- `cold` - The function is unlikely to be executed, so optimize it (and calls
  to it) differently.
- `naked` - The function utilizes a custom ABI or custom inline ASM that requires
  epilogue and prologue to be skipped.

## Static-only attributes

- `thread_local` - on a `static mut`, this signals that the value of this
  static may change depending on the current thread. The exact consequences of
  this are implementation-defined.

## FFI attributes

On an `extern` block, the following attributes are interpreted:

- `link_args` - specify arguments to the linker, rather than just the library
  name and type. This is feature gated and the exact behavior is
  implementation-defined (due to variety of linker invocation syntax).
- `link` - indicate that a native library should be linked to for the
  declarations in this block to be linked correctly. `link` supports an optional
  `kind` key with three possible values: `dylib`, `static`, and `framework`. See
  [external blocks](items.html#external-blocks) for more about external blocks. Two
  examples: `#[link(name = "readline")]` and
  `#[link(name = "CoreFoundation", kind = "framework")]`.
- `linked_from` - indicates what native library this block of FFI items is
  coming from. This attribute is of the form `#[linked_from = "foo"]` where
  `foo` is the name of a library in either `#[link]` or a `-l` flag. This
  attribute is currently required to export symbols from a Rust dynamic library
  on Windows, and it is feature gated behind the `linked_from` feature.

On declarations inside an `extern` block, the following attributes are
interpreted:

- `link_name` - the name of the symbol that this function or static should be
  imported as.
- `linkage` - on a static, this specifies the [linkage
  type](http://llvm.org/docs/LangRef.html#linkage-types).

On `enum`s:

- `repr` - on C-like enums, this sets the underlying type used for
  representation. Takes one argument, which is the primitive
  type this enum should be represented for, or `C`, which specifies that it
  should be the default `enum` size of the C ABI for that platform. Note that
  enum representation in C is undefined, and this may be incorrect when the C
  code is compiled with certain flags.

On `struct`s:

- `repr` - specifies the representation to use for this struct. Takes a list
  of options. The currently accepted ones are `C` and `packed`, which may be
  combined. `C` will use a C ABI compatible struct layout, and `packed` will
  remove any padding between fields (note that this is very fragile and may
  break platforms which require aligned access).

## Macro-related attributes

- `macro_use` on a `mod` — macros defined in this module will be visible in the
  module's parent, after this module has been included.

- `macro_use` on an `extern crate` — load macros from this crate.  An optional
  list of names `#[macro_use(foo, bar)]` restricts the import to just those
  macros named.  The `extern crate` must appear at the crate root, not inside
  `mod`, which ensures proper function of the [`$crate` macro
  variable](../book/macros.html#the-variable-crate).

- `macro_reexport` on an `extern crate` — re-export the named macros.

- `macro_export` - export a macro for cross-crate usage.

- `no_link` on an `extern crate` — even if we load this crate for macros, don't
  link it into the output.

See the [macros section of the
book](../book/macros.html#scoping-and-macro-importexport) for more information on
macro scope.

## Miscellaneous attributes

- `deprecated` - mark the item as deprecated; the full attribute is 
  `#[deprecated(since = "crate version", note = "...")`, where both arguments 
  are optional.
- `export_name` - on statics and functions, this determines the name of the
  exported symbol.
- `link_section` - on statics and functions, this specifies the section of the
  object file that this item's contents will be placed into.
- `no_mangle` - on any item, do not apply the standard name mangling. Set the
  symbol for this item to its identifier.
- `simd` - on certain tuple structs, derive the arithmetic operators, which
  lower to the target's SIMD instructions, if any; the `simd` feature gate
  is necessary to use this attribute.
- `unsafe_destructor_blind_to_params` - on `Drop::drop` method, asserts that the
  destructor code (and all potential specializations of that code) will
  never attempt to read from nor write to any references with lifetimes
  that come in via generic parameters. This is a constraint we cannot
  currently express via the type system, and therefore we rely on the
  programmer to assert that it holds. Adding this to a Drop impl causes
  the associated destructor to be considered "uninteresting" by the
  Drop-Check rule, and thus it can help sidestep data ordering
  constraints that would otherwise be introduced by the Drop-Check
  rule. Such sidestepping of the constraints, if done incorrectly, can
  lead to undefined behavior (in the form of reading or writing to data
  outside of its dynamic extent), and thus this attribute has the word
  "unsafe" in its name. To use this, the
  `unsafe_destructor_blind_to_params` feature gate must be enabled.
- `doc` - Doc comments such as `/// foo` are equivalent to `#[doc = "foo"]`.
- `rustc_on_unimplemented` - Write a custom note to be shown along with the error
   when the trait is found to be unimplemented on a type.
   You may use format arguments like `{T}`, `{A}` to correspond to the
   types at the point of use corresponding to the type parameters of the
   trait of the same name. `{Self}` will be replaced with the type that is supposed
   to implement the trait but doesn't. To use this, the `on_unimplemented` feature gate
   must be enabled.
- `must_use` - on structs and enums, will warn if a value of this type isn't used or
   assigned to a variable. You may also include an optional message by using
   `#[must_use = "message"]` which will be given alongside the warning.

### Conditional compilation

Sometimes one wants to have different compiler outputs from the same code,
depending on build target, such as targeted operating system, or to enable
release builds.

Configuration options are boolean (on or off) and are named either with a
single identifier (e.g. `foo`) or an identifier and a string (e.g. `foo = "bar"`;
the quotes are required and spaces around the `=` are unimportant). Note that
similarly-named options, such as `foo`, `foo="bar"` and `foo="baz"` may each be set
or unset independently.

Configuration options are either provided by the compiler or passed in on the
command line using `--cfg` (e.g. `rustc main.rs --cfg foo --cfg 'bar="baz"'`).
Rust code then checks for their presence using the `#[cfg(...)]` attribute:

```
// The function is only included in the build when compiling for OSX
#[cfg(target_os = "macos")]
fn macos_only() {
  // ...
}

// This function is only included when either foo or bar is defined
#[cfg(any(foo, bar))]
fn needs_foo_or_bar() {
  // ...
}

// This function is only included when compiling for a unixish OS with a 32-bit
// architecture
#[cfg(all(unix, target_pointer_width = "32"))]
fn on_32bit_unix() {
  // ...
}

// This function is only included when foo is not defined
#[cfg(not(foo))]
fn needs_not_foo() {
  // ...
}
```

This illustrates some conditional compilation can be achieved using the
`#[cfg(...)]` attribute. `any`, `all` and `not` can be used to assemble
arbitrarily complex configurations through nesting.

The following configurations must be defined by the implementation:

* `target_arch = "..."` - Target CPU architecture, such as `"x86"`,
  `"x86_64"` `"mips"`, `"powerpc"`, `"powerpc64"`, `"arm"`, or
  `"aarch64"`. This value is closely related to the first element of
  the platform target triple, though it is not identical.
* `target_os = "..."` - Operating system of the target, examples
  include `"windows"`, `"macos"`, `"ios"`, `"linux"`, `"android"`,
  `"freebsd"`, `"dragonfly"`, `"bitrig"` , `"openbsd"` or
  `"netbsd"`. This value is closely related to the second and third
  element of the platform target triple, though it is not identical.
* `target_family = "..."` - Operating system family of the target, e. g.
  `"unix"` or `"windows"`. The value of this configuration option is defined
  as a configuration itself, like `unix` or `windows`.
* `unix` - See `target_family`.
* `windows` - See `target_family`.
* `target_env = ".."` - Further disambiguates the target platform with
  information about the ABI/libc. Presently this value is either
  `"gnu"`, `"msvc"`, `"musl"`, or the empty string. For historical
  reasons this value has only been defined as non-empty when needed
  for disambiguation. Thus on many GNU platforms this value will be
  empty. This value is closely related to the fourth element of the
  platform target triple, though it is not identical. For example,
  embedded ABIs such as `gnueabihf` will simply define `target_env` as
  `"gnu"`.
* `target_endian = "..."` - Endianness of the target CPU, either `"little"` or
  `"big"`.
* `target_pointer_width = "..."` - Target pointer width in bits. This is set
  to `"32"` for targets with 32-bit pointers, and likewise set to `"64"` for
  64-bit pointers.
* `target_has_atomic = "..."` - Set of integer sizes on which the target can perform
  atomic operations. Values are `"8"`, `"16"`, `"32"`, `"64"` and `"ptr"`.
* `target_vendor = "..."` - Vendor of the target, for example `apple`, `pc`, or
  simply `"unknown"`.
* `test` - Enabled when compiling the test harness (using the `--test` flag).
* `debug_assertions` - Enabled by default when compiling without optimizations.
  This can be used to enable extra debugging code in development but not in
  production.  For example, it controls the behavior of the standard library's
  `debug_assert!` macro.

You can also set another attribute based on a `cfg` variable with `cfg_attr`:

```rust,ignore
#[cfg_attr(a, b)]
```

This is the same as `#[b]` if `a` is set by `cfg`, and nothing otherwise.

Lastly, configuration options can be used in expressions by invoking the `cfg!`
macro: `cfg!(a)` evaluates to `true` if `a` is set, and `false` otherwise.

### Lint check attributes

A lint check names a potentially undesirable coding pattern, such as
unreachable code or omitted documentation, for the static entity to which the
attribute applies.

For any lint check `C`:

* `allow(C)` overrides the check for `C` so that violations will go
   unreported,
* `deny(C)` signals an error after encountering a violation of `C`,
* `forbid(C)` is the same as `deny(C)`, but also forbids changing the lint
   level afterwards,
* `warn(C)` warns about violations of `C` but continues compilation.

The lint checks supported by the compiler can be found via `rustc -W help`,
along with their default settings.  [Compiler
plugins](../book/compiler-plugins.html#lint-plugins) can provide additional
lint checks.

```{.ignore}
pub mod m1 {
    // Missing documentation is ignored here
    #[allow(missing_docs)]
    pub fn undocumented_one() -> i32 { 1 }

    // Missing documentation signals a warning here
    #[warn(missing_docs)]
    pub fn undocumented_too() -> i32 { 2 }

    // Missing documentation signals an error here
    #[deny(missing_docs)]
    pub fn undocumented_end() -> i32 { 3 }
}
```

This example shows how one can use `allow` and `warn` to toggle a particular
check on and off:

```{.ignore}
#[warn(missing_docs)]
pub mod m2{
    #[allow(missing_docs)]
    pub mod nested {
        // Missing documentation is ignored here
        pub fn undocumented_one() -> i32 { 1 }

        // Missing documentation signals a warning here,
        // despite the allow above.
        #[warn(missing_docs)]
        pub fn undocumented_two() -> i32 { 2 }
    }

    // Missing documentation signals a warning here
    pub fn undocumented_too() -> i32 { 3 }
}
```

This example shows how one can use `forbid` to disallow uses of `allow` for
that lint check:

```{.ignore}
#[forbid(missing_docs)]
pub mod m3 {
    // Attempting to toggle warning signals an error here
    #[allow(missing_docs)]
    /// Returns 2.
    pub fn undocumented_too() -> i32 { 2 }
}
```

### Language items

Some primitive Rust operations are defined in Rust code, rather than being
implemented directly in C or assembly language. The definitions of these
operations have to be easy for the compiler to find. The `lang` attribute
makes it possible to declare these operations. For example, the `str` module
in the Rust standard library defines the string equality function:

```{.ignore}
#[lang = "str_eq"]
pub fn eq_slice(a: &str, b: &str) -> bool {
    // details elided
}
```

The name `str_eq` has a special meaning to the Rust compiler, and the presence
of this definition means that it will use this definition when generating calls
to the string equality function.

The set of language items is currently considered unstable. A complete
list of the built-in language items will be added in the future.

### Inline attributes

The inline attribute suggests that the compiler should place a copy of
the function or static in the caller, rather than generating code to
call the function or access the static where it is defined.

The compiler automatically inlines functions based on internal heuristics.
Incorrectly inlining functions can actually make the program slower, so it
should be used with care.

`#[inline]` and `#[inline(always)]` always cause the function to be serialized
into the crate metadata to allow cross-crate inlining.

There are three different types of inline attributes:

* `#[inline]` hints the compiler to perform an inline expansion.
* `#[inline(always)]` asks the compiler to always perform an inline expansion.
* `#[inline(never)]` asks the compiler to never perform an inline expansion.

### `derive`

The `derive` attribute allows certain traits to be automatically implemented
for data structures. For example, the following will create an `impl` for the
`PartialEq` and `Clone` traits for `Foo`, the type parameter `T` will be given
the `PartialEq` or `Clone` constraints for the appropriate `impl`:

```
#[derive(PartialEq, Clone)]
struct Foo<T> {
    a: i32,
    b: T,
}
```

The generated `impl` for `PartialEq` is equivalent to

```
# struct Foo<T> { a: i32, b: T }
impl<T: PartialEq> PartialEq for Foo<T> {
    fn eq(&self, other: &Foo<T>) -> bool {
        self.a == other.a && self.b == other.b
    }

    fn ne(&self, other: &Foo<T>) -> bool {
        self.a != other.a || self.b != other.b
    }
}
```

You can implement `derive` for your own type through [procedural
macros](procedural-macros.html).

### Compiler Features

Certain aspects of Rust may be implemented in the compiler, but they're not
necessarily ready for every-day use. These features are often of "prototype
quality" or "almost production ready", but may not be stable enough to be
considered a full-fledged language feature.

For this reason, Rust recognizes a special crate-level attribute of the form:

```{.ignore}
#![feature(feature1, feature2, feature3)]
```

This directive informs the compiler that the feature list: `feature1`,
`feature2`, and `feature3` should all be enabled. This is only recognized at a
crate-level, not at a module-level. Without this directive, all features are
considered off, and using the features will result in a compiler error.

The currently implemented features of the reference compiler are:

* `advanced_slice_patterns` - See the [match
                              expressions](expressions.html#match-expressions)
			      section for discussion; the exact semantics of
slice patterns are subject to change, so some types are still unstable.

* `slice_patterns` - OK, actually, slice patterns are just scary and
                     completely unstable.

* `asm` - The `asm!` macro provides a means for inline assembly. This is often
          useful, but the exact syntax for this feature along with its
          semantics are likely to change, so this macro usage must be opted
          into.

* `associated_consts` - Allows constants to be defined in `impl` and `trait`
                        blocks, so that they can be associated with a type or
                        trait in a similar manner to methods and associated
                        types.

* `box_patterns` - Allows `box` patterns, the exact semantics of which
                   is subject to change.

* `box_syntax` - Allows use of `box` expressions, the exact semantics of which
                 is subject to change.

* `cfg_target_vendor` - Allows conditional compilation using the `target_vendor`
                        matcher which is subject to change.

* `cfg_target_has_atomic` - Allows conditional compilation using the `target_has_atomic`
                            matcher which is subject to change.

* `concat_idents` - Allows use of the `concat_idents` macro, which is in many
                    ways insufficient for concatenating identifiers, and may be
                    removed entirely for something more wholesome.

* `custom_attribute` - Allows the usage of attributes unknown to the compiler
                       so that new attributes can be added in a backwards compatible
                       manner (RFC 572).

* `custom_derive` - Allows the use of `#[derive(Foo,Bar)]` as sugar for
                    `#[derive_Foo] #[derive_Bar]`, which can be user-defined syntax
                    extensions.

* `inclusive_range_syntax` - Allows use of the `a...b` and `...b` syntax for inclusive ranges.

* `inclusive_range` - Allows use of the types that represent desugared inclusive ranges.

* `intrinsics` - Allows use of the "rust-intrinsics" ABI. Compiler intrinsics
                 are inherently unstable and no promise about them is made.

* `lang_items` - Allows use of the `#[lang]` attribute. Like `intrinsics`,
                 lang items are inherently unstable and no promise about them
                 is made.

* `link_args` - This attribute is used to specify custom flags to the linker,
                but usage is strongly discouraged. The compiler's usage of the
                system linker is not guaranteed to continue in the future, and
                if the system linker is not used then specifying custom flags
                doesn't have much meaning.

* `link_llvm_intrinsics` - Allows linking to LLVM intrinsics via
                           `#[link_name="llvm.*"]`.

* `linkage` - Allows use of the `linkage` attribute, which is not portable.

* `log_syntax` - Allows use of the `log_syntax` macro attribute, which is a
                 nasty hack that will certainly be removed.

* `main` - Allows use of the `#[main]` attribute, which changes the entry point
           into a Rust program. This capability is subject to change.

* `macro_reexport` - Allows macros to be re-exported from one crate after being imported
                     from another. This feature was originally designed with the sole
                     use case of the Rust standard library in mind, and is subject to
                     change.

* `non_ascii_idents` - The compiler supports the use of non-ascii identifiers,
                       but the implementation is a little rough around the
                       edges, so this can be seen as an experimental feature
                       for now until the specification of identifiers is fully
                       fleshed out.

* `no_std` - Allows the `#![no_std]` crate attribute, which disables the implicit
             `extern crate std`. This typically requires use of the unstable APIs
             behind the libstd "facade", such as libcore and libcollections. It
             may also cause problems when using syntax extensions, including
             `#[derive]`.

* `on_unimplemented` - Allows the `#[rustc_on_unimplemented]` attribute, which allows
                       trait definitions to add specialized notes to error messages
                       when an implementation was expected but not found.

* `optin_builtin_traits` - Allows the definition of default and negative trait
                           implementations. Experimental.

* `plugin` - Usage of [compiler plugins][plugin] for custom lints or syntax extensions.
             These depend on compiler internals and are subject to change.

* `plugin_registrar` - Indicates that a crate provides [compiler plugins][plugin].

* `quote` - Allows use of the `quote_*!` family of macros, which are
            implemented very poorly and will likely change significantly
            with a proper implementation.

* `rustc_attrs` - Gates internal `#[rustc_*]` attributes which may be
                  for internal use only or have meaning added to them in the future.

* `rustc_diagnostic_macros`- A mysterious feature, used in the implementation
                             of rustc, not meant for mortals.

* `simd` - Allows use of the `#[simd]` attribute, which is overly simple and
           not the SIMD interface we want to expose in the long term.

* `simd_ffi` - Allows use of SIMD vectors in signatures for foreign functions.
               The SIMD interface is subject to change.

* `start` - Allows use of the `#[start]` attribute, which changes the entry point
            into a Rust program. This capability, especially the signature for the
            annotated function, is subject to change.

* `thread_local` - The usage of the `#[thread_local]` attribute is experimental
                   and should be seen as unstable. This attribute is used to
                   declare a `static` as being unique per-thread leveraging
                   LLVM's implementation which works in concert with the kernel
                   loader and dynamic linker. This is not necessarily available
                   on all platforms, and usage of it is discouraged.

* `trace_macros` - Allows use of the `trace_macros` macro, which is a nasty
                   hack that will certainly be removed.

* `unboxed_closures` - Rust's new closure design, which is currently a work in
                       progress feature with many known bugs.

* `allow_internal_unstable` - Allows `macro_rules!` macros to be tagged with the
                              `#[allow_internal_unstable]` attribute, designed
                              to allow `std` macros to call
                              `#[unstable]`/feature-gated functionality
                              internally without imposing on callers
                              (i.e. making them behave like function calls in
                              terms of encapsulation).

* `default_type_parameter_fallback` - Allows type parameter defaults to
                                      influence type inference.

* `stmt_expr_attributes` - Allows attributes on expressions.

* `type_ascription` - Allows type ascription expressions `expr: Type`.

* `abi_vectorcall` - Allows the usage of the vectorcall calling convention
                     (e.g. `extern "vectorcall" func fn_();`)

* `abi_sysv64` - Allows the usage of the system V AMD64 calling convention
                 (e.g. `extern "sysv64" func fn_();`)

If a feature is promoted to a language feature, then all existing programs will
start to receive compilation warnings about `#![feature]` directives which enabled
the new feature (because the directive is no longer necessary). However, if a
feature is decided to be removed from the language, errors will be issued (if
there isn't a parser error first). The directive in this case is no longer
necessary, and it's likely that existing code will break if the feature isn't
removed.

If an unknown feature is found in a directive, it results in a compiler error.
An unknown feature is one which has never been recognized by the compiler.
