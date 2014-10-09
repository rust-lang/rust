Version 0.12.0 (October 2014)
-----------------------------

  * ~1900 changes, numerous bugfixes

  * Highlights

    * The introductory documentation (now called The Rust Guide) has
      been completely rewritten, as have a number of supplementary
      guides.
    * Rust's package manager, Cargo, continues to improve and is
      sometimes considered to be quite awesome.
    * Many API's in `std` have been reviewed and updated for
      consistency with the in-development Rust coding
      guidelines. The standard library documentation tracks
      stabilization progress.
    * Minor libraries have been moved out-of-tree to the rust-lang org
      on GitHub: uuid, semver, glob, num, hexfloat, fourcc. They can
      be installed with Cargo.
    * Lifetime elision allows lifetime annotations to be left off of
      function declarations in many common scenarios.
    * Rust now works on 64-bit Windows.

  * Language
    * Indexing can be overloaded with the `Index` and `IndexMut`
      traits.
    * The `if let` construct takes a branch only if the `let` pattern
      matches, currently behind the 'if_let' feature gate.
    * 'where clauses', a more flexible syntax for specifying trait
      bounds that is more aesthetic, have been added for traits and
      free functions. Where clauses will in the future make it
      possible to constrain associated types, which would be
      impossible with the existing syntax.
    * A new slicing syntax (e.g. `[0..4]`) has been introduced behind
      the 'slicing_syntax' feature gate, and can be overloaded with
      the `Slice` or `SliceMut` traits.
    * The syntax for matching of sub-slices has been changed to use a
      postfix `..` instead of prefix (.e.g. `[a, b, c..]`), for
      consistency with other uses of `..` and to future-proof
      potential additional uses of the syntax.
    * The syntax for matching inclusive ranges in patterns has changed
      from `0..3` to `0...4` to be consistent with the exclusive range
      syntax for slicing.
    * Matching of sub-slices in non-tail positions (e.g.  `[a.., b,
      c]`) has been put behind the 'advanced_slice_patterns' feature
      gate and may be removed in the future.
    * Components of tuples and tuple structs can be extracted using
      the `value.0` syntax, currently behind the `tuple_indexing`
      feature gate.
    * The `#[crate_id]` attribute is no longer supported; versioning
      is handled by the package manager.
    * Renaming crate imports are now written `extern crate foo as bar`
      instead of `extern crate bar = foo`.
    * Renaming use statements are now written `use foo as bar` instead
      of `use bar = foo`.
    * `let` and `match` bindings and argument names in macros are now
      hygienic.
    * The new, more efficient, closure types ('unboxed closures') have
      been added under a feature gate, 'unboxed_closures'. These will
      soon replace the existing closure types, once higher-ranked
      trait lifetimes are added to the language.
    * `move` has been added as a keyword, for indicating closures
      that capture by value.
    * Mutation and assignment is no longer allowed in pattern guards.
    * Generic structs and enums can now have trait bounds.
    * The `Share` trait is now called `Sync` to free up the term
      'shared' to refer to 'shared reference' (the default reference
      type.
    * Dynamically-sized types have been mostly implemented,
      unifying the behavior of fat-pointer types with the rest of the
      type system.
    * As part of dynamically-sized types, the `Sized` trait has been
      introduced, which qualifying types implement by default, and
      which type parameters expect by default. To specify that a type
      parameter does not need to be sized, write `<Sized? T>`. Most
      types are `Sized`, notable exceptions being unsized arrays
      (`[T]`) and trait types.
    * Closures can return `!`, as in `|| -> !` or `proc() -> !`.
    * Lifetime bounds can now be applied to type parameters and object
      types.
    * The old, reference counted GC type, `Gc<T>` which was once
      denoted by the `@` sigil, has finally been removed. GC will be
      revisited in the future.

  * Libraries
    * Library documentation has been improved for a number of modules.
    * Bit-vectors, collections::bitv has been modernized.
    * The url crate is deprecated in favor of
      http://github.com/servo/rust-url, which can be installed with
      Cargo.
    * Most I/O stream types can be cloned and subsequently closed from
      a different thread.
    * A `std::time::Duration` type has been added for use in I/O
      methods that rely on timers, as well as in the 'time' crate's
      `Timespec` arithmetic.
    * The runtime I/O abstraction layer that enabled the green thread
      scheduler to do non-thread-blocking I/O has been removed, along
      with the libuv-based implementation employed by the green thread
      scheduler. This will greatly simplify the future I/O work.
    * `collections::btree` has been rewritten to have a more
      idiomatic and efficient design.

  * Tooling
    * rustdoc output now indicates the stability levels of API's.
    * The `--crate-name` flag can specify the name of the crate
      being compiled, like `#[crate_name]`.
    * The `-C metadata` specifies additional metadata to hash into
      symbol names, and `-C extra-filename` specifies additional
      information to put into the output filename, for use by the
      package manager for versioning.
    * debug info generation has continued to improve and should be
      more reliable under both gdb and lldb.
    * rustc has experimental support for compiling in parallel
      using the `-C codegen-units` flag.
    * rustc no longer encodes rpath information into binaries by
      default.

  * Misc
    * Stack usage has been optimized with LLVM lifetime annotations.
    * Official Rust binaries on Linux are more compatible with older
      kernels and distributions, built on CentOS 5.10.

Version 0.11.0 (July 2014)
-------------------------

  * ~1700 changes, numerous bugfixes

  * Language
    * ~[T] has been removed from the language. This type is superseded by
      the Vec<T> type.
    * ~str has been removed from the language. This type is superseded by
      the String type.
    * ~T has been removed from the language. This type is superseded by the
      Box<T> type.
    * @T has been removed from the language. This type is superseded by the
      standard library's std::gc::Gc<T> type.
    * Struct fields are now all private by default.
    * Vector indices and shift amounts are both required to be a `uint`
      instead of any integral type.
    * Byte character, byte string, and raw byte string literals are now all
      supported by prefixing the normal literal with a `b`.
    * Multiple ABIs are no longer allowed in an ABI string
    * The syntax for lifetimes on closures/procedures has been tweaked
      slightly: `<'a>|A, B|: 'b + K -> T`
    * Floating point modulus has been removed from the language; however it
      is still provided by a library implementation.
    * Private enum variants are now disallowed.
    * The `priv` keyword has been removed from the language.
    * A closure can no longer be invoked through a &-pointer.
    * The `use foo, bar, baz;` syntax has been removed from the language.
    * The transmute intrinsic no longer works on type parameters.
    * Statics now allow blocks/items in their definition.
    * Trait bounds are separated from objects with + instead of : now.
    * Objects can no longer be read while they are mutably borrowed.
    * The address of a static is now marked as insignificant unless the
      #[inline(never)] attribute is placed it.
    * The #[unsafe_destructor] attribute is now behind a feature gate.
    * Struct literals are no longer allowed in ambiguous positions such as
      if, while, match, and for..in.
    * Declaration of lang items and intrinsics are now feature-gated by
      default.
    * Integral literals no longer default to `int`, and floating point
      literals no longer default to `f64`. Literals must be suffixed with an
      appropriate type if inference cannot determine the type of the
      literal.
    * The Box<T> type is no longer implicitly borrowed to &mut T.
    * Procedures are now required to not capture borrowed references.

  * Libraries
    * The standard library is now a "facade" over a number of underlying
      libraries. This means that development on the standard library should
      be speeder due to smaller crates, as well as a clearer line between
      all dependencies.
    * A new library, libcore, lives under the standard library's facade
      which is Rust's "0-assumption" library, suitable for embedded and
      kernel development for example.
    * A regex crate has been added to the standard distribution. This crate
      includes statically compiled regular expressions.
    * The unwrap/unwrap_err methods on Result require a Show bound for
      better error messages.
    * The return types of the std::comm primitives have been centralized
      around the Result type.
    * A number of I/O primitives have gained the ability to time out their
      operations.
    * A number of I/O primitives have gained the ability to close their
      reading/writing halves to cancel pending operations.
    * Reverse iterator methods have been removed in favor of `rev()` on
      their forward-iteration counterparts.
    * A bitflags! macro has been added to enable easy interop with C and
      management of bit flags.
    * A debug_assert! macro is now provided which is disabled when
      `--cfg ndebug` is passed to the compiler.
    * A graphviz crate has been added for creating .dot files.
    * The std::cast module has been migrated into std::mem.
    * The std::local_data api has been migrated from freestanding functions
      to being based on methods.
    * The Pod trait has been renamed to Copy.
    * jemalloc has been added as the default allocator for types.
    * The API for allocating memory has been changed to use proper alignment
      and sized deallocation
    * Connecting a TcpStream or binding a TcpListener is now based on a
      string address and a u16 port. This allows connecting to a hostname as
      opposed to an IP.
    * The Reader trait now contains a core method, read_at_least(), which
      correctly handles many repeated 0-length reads.
    * The process-spawning API is now centered around a builder-style
      Command struct.
    * The :? printing qualifier has been moved from the standard library to
      an external libdebug crate.
    * Eq/Ord have been renamed to PartialEq/PartialOrd. TotalEq/TotalOrd
      have been renamed to Eq/Ord.
    * The select/plural methods have been removed from format!. The escapes
      for { and } have also changed from \{ and \} to {{ and }},
      respectively.
    * The TaskBuilder API has been re-worked to be a true builder, and
      extension traits for spawning native/green tasks have been added.

  * Tooling
    * All breaking changes to the language or libraries now have their
      commit message annotated with `[breaking-change]` to allow for easy
      discovery of breaking changes.
    * The compiler will now try to suggest how to annotate lifetimes if a
      lifetime-related error occurs.
    * Debug info continues to be improved greatly with general bug fixes and
      better support for situations like link time optimization (LTO).
    * Usage of syntax extensions when cross-compiling has been fixed.
    * Functionality equivalent to GCC & Clang's -ffunction-sections,
      -fdata-sections and --gc-sections has been enabled by default
    * The compiler is now stricter about where it will load module files
      from when a module is declared via `mod foo;`.
    * The #[phase(syntax)] attribute has been renamed to #[phase(plugin)].
      Syntax extensions are now discovered via a "plugin registrar" type
      which will be extended in the future to other various plugins.
    * Lints have been restructured to allow for dynamically loadable lints.
    * A number of rustdoc improvements:
      * The HTML output has been visually redesigned.
      * Markdown is now powered by hoedown instead of sundown.
      * Searching heuristics have been greatly improved.
      * The search index has been reduced in size by a great amount.
      * Cross-crate documentation via `pub use` has been greatly improved.
      * Primitive types are now hyperlinked and documented.
    * Documentation has been moved from static.rust-lang.org/doc to
      doc.rust-lang.org
    * A new sandbox, play.rust-lang.org, is available for running and
      sharing rust code examples on-line.
    * Unused attributes are now more robustly warned about.
    * The dead_code lint now warns about unused struct fields.
    * Cross-compiling to iOS is now supported.
    * Cross-compiling to mipsel is now supported.
    * Stability attributes are now inherited by default and no longer apply
      to intra-crate usage, only inter-crate usage.
    * Error message related to non-exhaustive match expressions have been
      greatly improved.

Version 0.10 (April 2014)
-------------------------

  * ~1500 changes, numerous bugfixes

  * Language
    * A new RFC process is now in place for modifying the language.
    * Patterns with `@`-pointers have been removed from the language.
    * Patterns with unique vectors (`~[T]`) have been removed from the
      language.
    * Patterns with unique strings (`~str`) have been removed from the
      language.
    * `@str` has been removed from the language.
    * `@[T]` has been removed from the language.
    * `@self` has been removed from the language.
    * `@Trait` has been removed from the language.
    * Headers on `~` allocations which contain `@` boxes inside the type for
      reference counting have been removed.
    * The semantics around the lifetimes of temporary expressions have changed,
      see #3511 and #11585 for more information.
    * Cross-crate syntax extensions are now possible, but feature gated. See
      #11151 for more information. This includes both `macro_rules!` macros as
      well as syntax extensions such as `format!`.
    * New lint modes have been added, and older ones have been turned on to be
      warn-by-default.
      * Unnecessary parentheses
      * Uppercase statics
      * Camel Case types
      * Uppercase variables
      * Publicly visible private types
      * `#[deriving]` with raw pointers
    * Unsafe functions can no longer be coerced to closures.
    * Various obscure macros such as `log_syntax!` are now behind feature gates.
    * The `#[simd]` attribute is now behind a feature gate.
    * Visibility is no longer allowed on `extern crate` statements, and
      unnecessary visibility (`priv`) is no longer allowed on `use` statements.
    * Trailing commas are now allowed in argument lists and tuple patterns.
    * The `do` keyword has been removed, it is now a reserved keyword.
    * Default type parameters have been implemented, but are feature gated.
    * Borrowed variables through captures in closures are now considered soundly.
    * `extern mod` is now `extern crate`
    * The `Freeze` trait has been removed.
    * The `Share` trait has been added for types that can be shared among
      threads.
    * Labels in macros are now hygienic.
    * Expression/statement macro invocations can be delimited with `{}` now.
    * Treatment of types allowed in `static mut` locations has been tweaked.
    * The `*` and `.` operators are now overloadable through the `Deref` and
      `DerefMut` traits.
    * `~Trait` and `proc` no longer have `Send` bounds by default.
    * Partial type hints are now supported with the `_` type marker.
    * An `Unsafe` type was introduced for interior mutability. It is now
      considered undefined to transmute from `&T` to `&mut T` without using the
      `Unsafe` type.
    * The #[linkage] attribute was implemented for extern statics/functions.
    * The inner attribute syntax has changed from `#[foo];` to `#![foo]`.
    * `Pod` was renamed to `Copy`.

  * Libraries
    * The `libextra` library has been removed. It has now been decomposed into
      component libraries with smaller and more focused nuggets of
      functionality. The full list of libraries can be found on the
      documentation index page.
    * std: `std::condition` has been removed. All I/O errors are now propagated
      through the `Result` type. In order to assist with error handling, a
      `try!` macro for unwrapping errors with an early return and a lint for
      unused results has been added. See #12039 for more information.
    * std: The `vec` module has been renamed to `slice`.
    * std: A new vector type, `Vec<T>`, has been added in preparation for DST.
      This will become the only growable vector in the future.
    * std: `std::io` now has more public-reexports. Types such as `BufferedReader`
      are now found at `std::io::BufferedReader` instead of
      `std::io::buffered::BufferedReader`.
    * std: `print` and `println` are no longer in the prelude, the `print!` and
      `println!` macros are intended to be used instead.
    * std: `Rc` now has a `Weak` pointer for breaking cycles, and it no longer
      attempts to statically prevent cycles.
    * std: The standard distribution is adopting the policy of pushing failure
      to the user rather than failing in libraries. Many functions (such as
      `slice::last()`) now return `Option<T>` instead of `T` + failing.
    * std: `fmt::Default` has been renamed to `fmt::Show`, and it now has a new
      deriving mode: `#[deriving(Show)]`.
    * std: `ToStr` is now implemented for all types implementing `Show`.
    * std: The formatting trait methods now take `&self` instead of `&T`
    * std: The `invert()` method on iterators has been renamed to `rev()`
    * std: `std::num` has seen a reduction in the genericity of its traits,
      consolidating functionality into a few core traits.
    * std: Backtraces are now printed on task failure if the environment
      variable `RUST_BACKTRACE` is present.
    * std: Naming conventions for iterators have been standardized. More details
      can be found on the wiki's style guide.
    * std: `eof()` has been removed from the `Reader` trait. Specific types may
      still implement the function.
    * std: Networking types are now cloneable to allow simultaneous reads/writes.
    * std: `assert_approx_eq!` has been removed
    * std: The `e` and `E` formatting specifiers for floats have been added to
      print them in exponential notation.
    * std: The `Times` trait has been removed
    * std: Indications of variance and opting out of builtin bounds is done
      through marker types in `std::kinds::marker` now
    * std: `hash` has been rewritten, `IterBytes` has been removed, and
      `#[deriving(Hash)]` is now possible.
    * std: `SharedChan` has been removed, `Sender` is now cloneable.
    * std: `Chan` and `Port` were renamed to `Sender` and `Receiver`.
    * std: `Chan::new` is now `channel()`.
    * std: A new synchronous channel type has been implemented.
    * std: A `select!` macro is now provided for selecting over `Receiver`s.
    * std: `hashmap` and `trie` have been moved to `libcollections`
    * std: `run` has been rolled into `io::process`
    * std: `assert_eq!` now uses `{}` instead of `{:?}`
    * std: The equality and comparison traits have seen some reorganization.
    * std: `rand` has moved to `librand`.
    * std: `to_{lower,upper}case` has been implemented for `char`.
    * std: Logging has been moved to `liblog`.
    * collections: `HashMap` has been rewritten for higher performance and less
      memory usage.
    * native: The default runtime is now `libnative`. If `libgreen` is desired,
      it can be booted manually. The runtime guide has more information and
      examples.
    * native: All I/O functionality except signals has been implemented.
    * green: Task spawning with `libgreen` has been optimized with stack caching
      and various trimming of code.
    * green: Tasks spawned by `libgreen` now have an unmapped guard page.
    * sync: The `extra::sync` module has been updated to modern rust (and moved
      to the `sync` library), tweaking and improving various interfaces while
      dropping redundant functionality.
    * sync: A new `Barrier` type has been added to the `sync` library.
    * sync: An efficient mutex for native and green tasks has been implemented.
    * serialize: The `base64` module has seen some improvement. It treats
      newlines better, has non-string error values, and has seen general
      cleanup.
    * fourcc: A `fourcc!` macro was introduced
    * hexfloat: A `hexfloat!` macro was implemented for specifying floats via a
      hexadecimal literal.

  * Tooling
    * `rustpkg` has been deprecated and removed from the main repository. Its
      replacement, `cargo`, is under development.
    * Nightly builds of rust are now available
    * The memory usage of rustc has been improved many times throughout this
      release cycle.
    * The build process supports disabling rpath support for the rustc binary
      itself.
    * Code generation has improved in some cases, giving more information to the
      LLVM optimization passes to enable more extensive optimizations.
    * Debuginfo compatibility with lldb on OSX has been restored.
    * The master branch is now gated on an android bot, making building for
      android much more reliable.
    * Output flags have been centralized into one `--emit` flag.
    * Crate type flags have been centralized into one `--crate-type` flag.
    * Codegen flags have been consolidated behind a `-C` flag.
    * Linking against outdated crates now has improved error messages.
    * Error messages with lifetimes will often suggest how to annotate the
      function to fix the error.
    * Many more types are documented in the standard library, and new guides
      were written.
    * Many `rustdoc` improvements:
      * code blocks are syntax highlighted.
      * render standalone markdown files.
      * the --test flag tests all code blocks by default.
      * exported macros are displayed.
      * reexported types have their documentation inlined at the location of the
        first reexport.
      * search works across crates that have been rendered to the same output
        directory.

Version 0.9 (January 2014)
--------------------------

   * ~1800 changes, numerous bugfixes

   * Language
      * The `float` type has been removed. Use `f32` or `f64` instead.
      * A new facility for enabling experimental features (feature gating) has
        been added, using the crate-level `#[feature(foo)]` attribute.
      * Managed boxes (@) are now behind a feature gate
        (`#[feature(managed_boxes)]`) in preparation for future removal. Use the
        standard library's `Gc` or `Rc` types instead.
      * `@mut` has been removed. Use `std::cell::{Cell, RefCell}` instead.
      * Jumping back to the top of a loop is now done with `continue` instead of
        `loop`.
      * Strings can no longer be mutated through index assignment.
      * Raw strings can be created via the basic `r"foo"` syntax or with matched
        hash delimiters, as in `r###"foo"###`.
      * `~fn` is now written `proc (args) -> retval { ... }` and may only be
        called once.
      * The `&fn` type is now written `|args| -> ret` to match the literal form.
      * `@fn`s have been removed.
      * `do` only works with procs in order to make it obvious what the cost
        of `do` is.
      * Single-element tuple-like structs can no longer be dereferenced to
        obtain the inner value. A more comprehensive solution for overloading
        the dereference operator will be provided in the future.
      * The `#[link(...)]` attribute has been replaced with
        `#[crate_id = "name#vers"]`.
      * Empty `impl`s must be terminated with empty braces and may not be
        terminated with a semicolon.
      * Keywords are no longer allowed as lifetime names; the `self` lifetime
        no longer has any special meaning.
      * The old `fmt!` string formatting macro has been removed.
      * `printf!` and `printfln!` (old-style formatting) removed in favor of
        `print!` and `println!`.
      * `mut` works in patterns now, as in `let (mut x, y) = (1, 2);`.
      * The `extern mod foo (name = "bar")` syntax has been removed. Use
        `extern mod foo = "bar"` instead.
      * New reserved keywords: `alignof`, `offsetof`, `sizeof`.
      * Macros can have attributes.
      * Macros can expand to items with attributes.
      * Macros can expand to multiple items.
      * The `asm!` macro is feature-gated (`#[feature(asm)]`).
      * Comments may be nested.
      * Values automatically coerce to trait objects they implement, without
        an explicit `as`.
      * Enum discriminants are no longer an entire word but as small as needed to
        contain all the variants. The `repr` attribute can be used to override
        the discriminant size, as in `#[repr(int)]` for integer-sized, and
        `#[repr(C)]` to match C enums.
      * Non-string literals are not allowed in attributes (they never worked).
      * The FFI now supports variadic functions.
      * Octal numeric literals, as in `0o7777`.
      * The `concat!` syntax extension performs compile-time string concatenation.
      * The `#[fixed_stack_segment]` and `#[rust_stack]` attributes have been
        removed as Rust no longer uses segmented stacks.
      * Non-ascii identifiers are feature-gated (`#[feature(non_ascii_idents)]`).
      * Ignoring all fields of an enum variant or tuple-struct is done with `..`,
        not `*`; ignoring remaining fields of a struct is also done with `..`,
        not `_`; ignoring a slice of a vector is done with `..`, not `.._`.
      * `rustc` supports the "win64" calling convention via `extern "win64"`.
      * `rustc` supports the "system" calling convention, which defaults to the
        preferred convention for the target platform, "stdcall" on 32-bit Windows,
        "C" elsewhere.
      * The `type_overflow` lint (default: warn) checks literals for overflow.
      * The `unsafe_block` lint (default: allow) checks for usage of `unsafe`.
      * The `attribute_usage` lint (default: warn) warns about unknown
        attributes.
      * The `unknown_features` lint (default: warn) warns about unknown
        feature gates.
      * The `dead_code` lint (default: warn) checks for dead code.
      * Rust libraries can be linked statically to one another
      * `#[link_args]` is behind the `link_args` feature gate.
      * Native libraries are now linked with `#[link(name = "foo")]`
      * Native libraries can be statically linked to a rust crate
        (`#[link(name = "foo", kind = "static")]`).
      * Native OS X frameworks are now officially supported
        (`#[link(name = "foo", kind = "framework")]`).
      * The `#[thread_local]` attribute creates thread-local (not task-local)
        variables. Currently behind the `thread_local` feature gate.
      * The `return` keyword may be used in closures.
      * Types that can be copied via a memcpy implement the `Pod` kind.
      * The `cfg` attribute can now be used on struct fields and enum variants.

   * Libraries
      * std: The `option` and `result` API's have been overhauled to make them
        simpler, more consistent, and more composable.
      * std: The entire `std::io` module has been replaced with one that is
        more comprehensive and that properly interfaces with the underlying
        scheduler. File, TCP, UDP, Unix sockets, pipes, and timers are all
        implemented.
      * std: `io::util` contains a number of useful implementations of
        `Reader` and `Writer`, including `NullReader`, `NullWriter`,
        `ZeroReader`, `TeeReader`.
      * std: The reference counted pointer type `extra::rc` moved into std.
      * std: The `Gc` type in the `gc` module will replace `@` (it is currently
        just a wrapper around it).
      * std: The `Either` type has been removed.
      * std: `fmt::Default` can be implemented for any type to provide default
        formatting to the `format!` macro, as in `format!("{}", myfoo)`.
      * std: The `rand` API continues to be tweaked.
      * std: The `rust_begin_unwind` function, useful for inserting breakpoints
        on failure in gdb, is now named `rust_fail`.
      * std: The `each_key` and `each_value` methods on `HashMap` have been
        replaced by the `keys` and `values` iterators.
      * std: Functions dealing with type size and alignment have moved from the
        `sys` module to the `mem` module.
      * std: The `path` module was written and API changed.
      * std: `str::from_utf8` has been changed to cast instead of allocate.
      * std: `starts_with` and `ends_with` methods added to vectors via the
        `ImmutableEqVector` trait, which is in the prelude.
      * std: Vectors can be indexed with the `get_opt` method, which returns `None`
        if the index is out of bounds.
      * std: Task failure no longer propagates between tasks, as the model was
        complex, expensive, and incompatible with thread-based tasks.
      * std: The `Any` type can be used for dynamic typing.
      * std: `~Any` can be passed to the `fail!` macro and retrieved via
        `task::try`.
      * std: Methods that produce iterators generally do not have an `_iter`
        suffix now.
      * std: `cell::Cell` and `cell::RefCell` can be used to introduce mutability
        roots (mutable fields, etc.). Use instead of e.g. `@mut`.
      * std: `util::ignore` renamed to `prelude::drop`.
      * std: Slices have `sort` and `sort_by` methods via the `MutableVector`
        trait.
      * std: `vec::raw` has seen a lot of cleanup and API changes.
      * std: The standard library no longer includes any C++ code, and very
        minimal C, eliminating the dependency on libstdc++.
      * std: Runtime scheduling and I/O functionality has been factored out into
        extensible interfaces and is now implemented by two different crates:
        libnative, for native threading and I/O; and libgreen, for green threading
        and I/O. This paves the way for using the standard library in more limited
        embedded environments.
      * std: The `comm` module has been rewritten to be much faster, have a
        simpler, more consistent API, and to work for both native and green
        threading.
      * std: All libuv dependencies have been moved into the rustuv crate.
      * native: New implementations of runtime scheduling on top of OS threads.
      * native: New native implementations of TCP, UDP, file I/O, process spawning,
        and other I/O.
      * green: The green thread scheduler and message passing types are almost
        entirely lock-free.
      * extra: The `flatpipes` module had bitrotted and was removed.
      * extra: All crypto functions have been removed and Rust now has a policy of
        not reimplementing crypto in the standard library. In the future crypto
        will be provided by external crates with bindings to established libraries.
      * extra: `c_vec` has been modernized.
      * extra: The `sort` module has been removed. Use the `sort` method on
        mutable slices.

   * Tooling
      * The `rust` and `rusti` commands have been removed, due to lack of
        maintenance.
      * `rustdoc` was completely rewritten.
      * `rustdoc` can test code examples in documentation.
      * `rustpkg` can test packages with the argument, 'test'.
      * `rustpkg` supports arbitrary dependencies, including C libraries.
      * `rustc`'s support for generating debug info is improved again.
      * `rustc` has better error reporting for unbalanced delimiters.
      * `rustc`'s JIT support was removed due to bitrot.
      * Executables and static libraries can be built with LTO (-Z lto)
      * `rustc` adds a `--dep-info` flag for communicating dependencies to
        build tools.

Version 0.8 (September 2013)
--------------------------

   * ~2200 changes, numerous bugfixes

   * Language
      * The `for` loop syntax has changed to work with the `Iterator` trait.
      * At long last, unwinding works on Windows.
      * Default methods are ready for use.
      * Many trait inheritance bugs fixed.
      * Owned and borrowed trait objects work more reliably.
      * `copy` is no longer a keyword. It has been replaced by the `Clone` trait.
      * rustc can omit emission of code for the `debug!` macro if it is passed
        `--cfg ndebug`
      * mod.rs is now "blessed". When loading `mod foo;`, rustc will now look
        for foo.rs, then foo/mod.rs, and will generate an error when both are
        present.
      * Strings no longer contain trailing nulls. The new `std::c_str` module
        provides new mechanisms for converting to C strings.
      * The type of foreign functions is now `extern "C" fn` instead of `*u8'.
      * The FFI has been overhauled such that foreign functions are called directly,
        instead of through a stack-switching wrapper.
      * Calling a foreign function must be done through a Rust function with the
        `#[fixed_stack_segment]` attribute.
      * The `externfn!` macro can be used to declare both a foreign function and
        a `#[fixed_stack_segment]` wrapper at once.
      * `pub` and `priv` modifiers on `extern` blocks are no longer parsed.
      * `unsafe` is no longer allowed on extern fns - they are all unsafe.
      * `priv` is disallowed everywhere except for struct fields and enum variants.
      * `&T` (besides `&'static T`) is no longer allowed in `@T`.
      * `ref` bindings in irrefutable patterns work correctly now.
      * `char` is now prevented from containing invalid code points.
      * Casting to `bool` is no longer allowed.
      * `\0` is now accepted as an escape in chars and strings.
      * `yield` is a reserved keyword.
      * `typeof` is a reserved keyword.
      * Crates may be imported by URL with `extern mod foo = "url";`.
      * Explicit enum discriminants may be given as uints as in `enum E { V = 0u }`
      * Static vectors can be initialized with repeating elements,
        e.g. `static foo: [u8, .. 100]: [0, .. 100];`.
      * Static structs can be initialized with functional record update,
        e.g. `static foo: Foo = Foo { a: 5, .. bar };`.
      * `cfg!` can be used to conditionally execute code based on the crate
        configuration, similarly to `#[cfg(...)]`.
      * The `unnecessary_qualification` lint detects unneeded module
        prefixes (default: allow).
      * Arithmetic operations have been implemented on the SIMD types in
        `std::unstable::simd`.
      * Exchange allocation headers were removed, reducing memory usage.
      * `format!` implements a completely new, extensible, and higher-performance
        string formatting system. It will replace `fmt!`.
      * `print!` and `println!` write formatted strings (using the `format!`
        extension) to stdout.
      * `write!` and `writeln!` write formatted strings (using the `format!`
        extension) to the new Writers in `std::rt::io`.
      * The library section in which a function or static is placed may
        be specified with `#[link_section = "..."]`.
      * The `proto!` syntax extension for defining bounded message protocols
        was removed.
      * `macro_rules!` is hygienic for `let` declarations.
      * The `#[export_name]` attribute specifies the name of a symbol.
      * `unreachable!` can be used to indicate unreachable code, and fails
        if executed.

   * Libraries
      * std: Transitioned to the new runtime, written in Rust.
      * std: Added an experimental I/O library, `rt::io`, based on the new
        runtime.
      * std: A new generic `range` function was added to the prelude, replacing
        `uint::range` and friends.
      * std: `range_rev` no longer exists. Since range is an iterator it can be
        reversed with `range(lo, hi).invert()`.
      * std: The `chain` method on option renamed to `and_then`; `unwrap_or_default`
        renamed to `unwrap_or`.
      * std: The `iterator` module was renamed to `iter`.
      * std: Integral types now support the `checked_add`, `checked_sub`, and
        `checked_mul` operations for detecting overflow.
      * std: Many methods in `str`, `vec`, `option, `result` were renamed for
        consistency.
      * std: Methods are standardizing on conventions for casting methods:
        `to_foo` for copying, `into_foo` for moving, `as_foo` for temporary
        and cheap casts.
      * std: The `CString` type in `c_str` provides new ways to convert to and
        from C strings.
      * std: `DoubleEndedIterator` can yield elements in two directions.
      * std: The `mut_split` method on vectors partitions an `&mut [T]` into
        two splices.
      * std: `str::from_bytes` renamed to `str::from_utf8`.
      * std: `pop_opt` and `shift_opt` methods added to vectors.
      * std: The task-local data interface no longer uses @, and keys are
        no longer function pointers.
      * std: The `swap_unwrap` method of `Option` renamed to `take_unwrap`.
      * std: Added `SharedPort` to `comm`.
      * std: `Eq` has a default method for `ne`; only `eq` is required
        in implementations.
      * std: `Ord` has default methods for `le`, `gt` and `ge`; only `lt`
        is required in implementations.
      * std: `is_utf8` performance is improved, impacting many string functions.
      * std: `os::MemoryMap` provides cross-platform mmap.
      * std: `ptr::offset` is now unsafe, but also more optimized. Offsets that
        are not 'in-bounds' are considered undefined.
      * std: Many freestanding functions in `vec` removed in favor of methods.
      * std: Many freestanding functions on scalar types removed in favor of
        methods.
      * std: Many options to task builders were removed since they don't make
        sense in the new scheduler design.
      * std: More containers implement `FromIterator` so can be created by the
        `collect` method.
      * std: More complete atomic types in `unstable::atomics`.
      * std: `comm::PortSet` removed.
      * std: Mutating methods in the `Set` and `Map` traits have been moved into
        the `MutableSet` and `MutableMap` traits. `Container::is_empty`,
        `Map::contains_key`, `MutableMap::insert`, and `MutableMap::remove` have
        default implementations.
      * std: Various `from_str` functions were removed in favor of a generic
        `from_str` which is available in the prelude.
      * std: `util::unreachable` removed in favor of the `unreachable!` macro.
      * extra: `dlist`, the doubly-linked list was modernized.
      * extra: Added a `hex` module with `ToHex` and `FromHex` traits.
      * extra: Added `glob` module, replacing `std::os::glob`.
      * extra: `rope` was removed.
      * extra: `deque` was renamed to `ringbuf`. `RingBuf` implements `Deque`.
      * extra: `net`, and `timer` were removed. The experimental replacements
        are `std::rt::io::net` and `std::rt::io::timer`.
      * extra: Iterators implemented for `SmallIntMap`.
      * extra: Iterators implemented for `Bitv` and `BitvSet`.
      * extra: `SmallIntSet` removed. Use `BitvSet`.
      * extra: Performance of JSON parsing greatly improved.
      * extra: `semver` updated to SemVer 2.0.0.
      * extra: `term` handles more terminals correctly.
      * extra: `dbg` module removed.
      * extra: `par` module removed.
      * extra: `future` was cleaned up, with some method renames.
      * extra: Most free functions in `getopts` were converted to methods.

   * Other
      * rustc's debug info generation (`-Z debug-info`) is greatly improved.
      * rustc accepts `--target-cpu` to compile to a specific CPU architecture,
        similarly to gcc's `--march` flag.
      * rustc's performance compiling small crates is much better.
      * rustpkg has received many improvements.
      * rustpkg supports git tags as package IDs.
      * rustpkg builds into target-specific directories so it can be used for
        cross-compiling.
      * The number of concurrent test tasks is controlled by the environment
        variable RUST_TEST_TASKS.
      * The test harness can now report metrics for benchmarks.
      * All tools have man pages.
      * Programs compiled with `--test` now support the `-h` and `--help` flags.
      * The runtime uses jemalloc for allocations.
      * Segmented stacks are temporarily disabled as part of the transition to
        the new runtime. Stack overflows are possible!
      * A new documentation backend, rustdoc_ng, is available for use. It is
        still invoked through the normal `rustdoc` command.

Version 0.7 (July 2013)
-----------------------

   * ~2000 changes, numerous bugfixes

   * Language
      * `impl`s no longer accept a visibility qualifier. Put them on methods
        instead.
      * The borrow checker has been rewritten with flow-sensitivity, fixing
        many bugs and inconveniences.
      * The `self` parameter no longer implicitly means `&'self self`,
        and can be explicitly marked with a lifetime.
      * Overloadable compound operators (`+=`, etc.) have been temporarily
        removed due to bugs.
      * The `for` loop protocol now requires `for`-iterators to return `bool`
        so they compose better.
      * The `Durable` trait is replaced with the `'static` bounds.
      * Trait default methods work more often.
      * Structs with the `#[packed]` attribute have byte alignment and
        no padding between fields.
      * Type parameters bound by `Copy` must now be copied explicitly with
        the `copy` keyword.
      * It is now illegal to move out of a dereferenced unsafe pointer.
      * `Option<~T>` is now represented as a nullable pointer.
      * `@mut` does dynamic borrow checks correctly.
      * The `main` function is only detected at the topmost level of the crate.
        The `#[main]` attribute is still valid anywhere.
      * Struct fields may no longer be mutable. Use inherited mutability.
      * The `#[no_send]` attribute makes a type that would otherwise be
        `Send`, not.
      * The `#[no_freeze]` attribute makes a type that would otherwise be
        `Freeze`, not.
      * Unbounded recursion will abort the process after reaching the limit
        specified by the `RUST_MAX_STACK` environment variable (default: 1GB).
      * The `vecs_implicitly_copyable` lint mode has been removed. Vectors
        are never implicitly copyable.
      * `#[static_assert]` makes compile-time assertions about static bools.
      * At long last, 'argument modes' no longer exist.
      * The rarely used `use mod` statement no longer exists.

   * Syntax extensions
      * `fail!` and `assert!` accept `~str`, `&'static str` or `fmt!`-style
        argument list.
      * `Encodable`, `Decodable`, `Ord`, `TotalOrd`, `TotalEq`, `DeepClone`,
        `Rand`, `Zero` and `ToStr` can all be automatically derived with
        `#[deriving(...)]`.
      * The `bytes!` macro returns a vector of bytes for string, u8, char,
        and unsuffixed integer literals.

   * Libraries
      * The `core` crate was renamed to `std`.
      * The `std` crate was renamed to `extra`.
      * More and improved documentation.
      * std: `iterator` module for external iterator objects.
      * Many old-style (internal, higher-order function) iterators replaced by
        implementations of `Iterator`.
      * std: Many old internal vector and string iterators,
        incl. `any`, `all`. removed.
      * std: The `finalize` method of `Drop` renamed to `drop`.
      * std: The `drop` method now takes `&mut self` instead of `&self`.
      * std: The prelude no longer reexports any modules, only types and traits.
      * std: Prelude additions: `print`, `println`, `FromStr`, `ApproxEq`, `Equiv`,
        `Iterator`, `IteratorUtil`, many numeric traits, many tuple traits.
      * std: New numeric traits: `Fractional`, `Real`, `RealExt`, `Integer`, `Ratio`,
        `Algebraic`, `Trigonometric`, `Exponential`, `Primitive`.
      * std: Tuple traits and accessors defined for up to 12-tuples, e.g.
        `(0, 1, 2).n2()` or `(0, 1, 2).n2_ref()`.
      * std: Many types implement `Clone`.
      * std: `path` type renamed to `Path`.
      * std: `mut` module and `Mut` type removed.
      * std: Many standalone functions removed in favor of methods and iterators
        in `vec`, `str`. In the future methods will also work as functions.
      * std: `reinterpret_cast` removed. Use `transmute`.
      * std: ascii string handling in `std::ascii`.
      * std: `Rand` is implemented for ~/@.
      * std: `run` module for spawning processes overhauled.
      * std: Various atomic types added to `unstable::atomic`.
      * std: Various types implement `Zero`.
      * std: `LinearMap` and `LinearSet` renamed to `HashMap` and `HashSet`.
      * std: Borrowed pointer functions moved from `ptr` to `borrow`.
      * std: Added `os::mkdir_recursive`.
      * std: Added `os::glob` function performs filesystems globs.
      * std: `FuzzyEq` renamed to `ApproxEq`.
      * std: `Map` now defines `pop` and `swap` methods.
      * std: `Cell` constructors converted to static methods.
      * extra: `rc` module adds the reference counted pointers, `Rc` and `RcMut`.
      * extra: `flate` module moved from `std` to `extra`.
      * extra: `fileinput` module for iterating over a series of files.
      * extra: `Complex` number type and `complex` module.
      * extra: `Rational` number type and `rational` module.
      * extra: `BigInt`, `BigUint` implement numeric and comparison traits.
      * extra: `term` uses terminfo now, is more correct.
      * extra: `arc` functions converted to methods.
      * extra: Implementation of fixed output size variations of SHA-2.

   * Tooling
      * `unused_variable`  lint mode for unused variables (default: warn).
      * `unused_unsafe` lint mode for detecting unnecessary `unsafe` blocks
        (default: warn).
      * `unused_mut` lint mode for identifying unused `mut` qualifiers
        (default: warn).
      * `dead_assignment` lint mode for unread variables (default: warn).
      * `unnecessary_allocation` lint mode detects some heap allocations that are
        immediately borrowed so could be written without allocating (default: warn).
      * `missing_doc` lint mode (default: allow).
      * `unreachable_code` lint mode (default: warn).
      * The `rusti` command has been rewritten and a number of bugs addressed.
      * rustc outputs in color on more terminals.
      * rustc accepts a `--link-args` flag to pass arguments to the linker.
      * rustc accepts a `-Z print-link-args` flag for debugging linkage.
      * Compiling with `-g` will make the binary record information about
        dynamic borrowcheck failures for debugging.
      * rustdoc has a nicer stylesheet.
      * Various improvements to rustdoc.
      * Improvements to rustpkg (see the detailed release notes).

Version 0.6 (April 2013)
------------------------

   * ~2100 changes, numerous bugfixes

   * Syntax changes
      * The self type parameter in traits is now spelled `Self`
      * The `self` parameter in trait and impl methods must now be explicitly
        named (for example: `fn f(&self) { }`). Implicit self is deprecated.
      * Static methods no longer require the `static` keyword and instead
        are distinguished by the lack of a `self` parameter
      * Replaced the `Durable` trait with the `'static` lifetime
      * The old closure type syntax with the trailing sigil has been
        removed in favor of the more consistent leading sigil
      * `super` is a keyword, and may be prefixed to paths
      * Trait bounds are separated with `+` instead of whitespace
      * Traits are implemented with `impl Trait for Type`
        instead of `impl Type: Trait`
      * Lifetime syntax is now `&'l foo` instead of `&l/foo`
      * The `export` keyword has finally been removed
      * The `move` keyword has been removed (see "Semantic changes")
      * The interior mutability qualifier on vectors, `[mut T]`, has been
        removed. Use `&mut [T]`, etc.
      * `mut` is no longer valid in `~mut T`. Use inherited mutability
      * `fail` is no longer a keyword. Use `fail!()`
      * `assert` is no longer a keyword. Use `assert!()`
      * `log` is no longer a keyword. use `debug!`, etc.
      * 1-tuples may be represented as `(T,)`
      * Struct fields may no longer be `mut`. Use inherited mutability,
        `@mut T`, `core::mut` or `core::cell`
      * `extern mod { ... }` is no longer valid syntax for foreign
        function modules. Use extern blocks: `extern { ... }`
      * Newtype enums removed. Use tuple-structs.
      * Trait implementations no longer support visibility modifiers
      * Pattern matching over vectors improved and expanded
      * `const` renamed to `static` to correspond to lifetime name,
        and make room for future `static mut` unsafe mutable globals.
      * Replaced `#[deriving_eq]` with `#[deriving(Eq)]`, etc.
      * `Clone` implementations can be automatically generated with
        `#[deriving(Clone)]`
      * Casts to traits must use a pointer sigil, e.g. `@foo as @Bar`
        instead of `foo as Bar`.
      * Fixed length vector types are now written as `[int, .. 3]`
        instead of `[int * 3]`.
      * Fixed length vector types can express the length as a constant
        expression. (ex: `[int, .. GL_BUFFER_SIZE - 2]`)

   * Semantic changes
      * Types with owned pointers or custom destructors move by default,
        eliminating the `move` keyword
      * All foreign functions are considered unsafe
      * &mut is now unaliasable
      * Writes to borrowed @mut pointers are prevented dynamically
      * () has size 0
      * The name of the main function can be customized using #[main]
      * The default type of an inferred closure is &fn instead of @fn
      * `use` statements may no longer be "chained" - they cannot import
        identifiers imported by previous `use` statements
      * `use` statements are crate relative, importing from the "top"
        of the crate by default. Paths may be prefixed with `super::`
        or `self::` to change the search behavior.
      * Method visibility is inherited from the implementation declaration
      * Structural records have been removed
      * Many more types can be used in static items, including enums
        'static-lifetime pointers and vectors
      * Pattern matching over vectors improved and expanded
      * Typechecking of closure types has been overhauled to
        improve inference and eliminate unsoundness
      * Macros leave scope at the end of modules, unless that module is
        tagged with #[macro_escape]

   * Libraries
      * Added big integers to `std::bigint`
      * Removed `core::oldcomm` module
      * Added pipe-based `core::comm` module
      * Numeric traits have been reorganized under `core::num`
      * `vec::slice` finally returns a slice
      * `debug!` and friends don't require a format string, e.g. `debug!(Foo)`
      * Containers reorganized around traits in `core::container`
      * `core::dvec` removed, `~[T]` is a drop-in replacement
      * `core::send_map` renamed to `core::hashmap`
      * `std::map` removed; replaced with `core::hashmap`
      * `std::treemap` reimplemented as an owned balanced tree
      * `std::deque` and `std::smallintmap` reimplemented as owned containers
      * `core::trie` added as a fast ordered map for integer keys
      * Set types added to `core::hashmap`, `core::trie` and `std::treemap`
      * `Ord` split into `Ord` and `TotalOrd`. `Ord` is still used to
        overload the comparison operators, whereas `TotalOrd` is used
        by certain container types

   * Other
      * Replaced the 'cargo' package manager with 'rustpkg'
      * Added all-purpose 'rust' tool
      * `rustc --test` now supports benchmarks with the `#[bench]` attribute
      * rustc now *attempts* to offer spelling suggestions
      * Improved support for ARM and Android
      * Preliminary MIPS backend
      * Improved foreign function ABI implementation for x86, x86_64
      * Various memory usage improvements
      * Rust code may be embedded in foreign code under limited circumstances
      * Inline assembler supported by new asm!() syntax extension.

Version 0.5 (December 2012)
---------------------------

   * ~900 changes, numerous bugfixes

   * Syntax changes
      * Removed `<-` move operator
      * Completed the transition from the `#fmt` extension syntax to `fmt!`
      * Removed old fixed length vector syntax - `[T]/N`
      * New token-based quasi-quoters, `quote_tokens!`, `quote_expr!`, etc.
      * Macros may now expand to items and statements
      * `a.b()` is always parsed as a method call, never as a field projection
      * `Eq` and `IterBytes` implementations can be automatically generated
        with `#[deriving_eq]` and `#[deriving_iter_bytes]` respectively
      * Removed the special crate language for `.rc` files
      * Function arguments may consist of any irrefutable pattern

   * Semantic changes
      * `&` and `~` pointers may point to objects
      * Tuple structs - `struct Foo(Bar, Baz)`. Will replace newtype enums.
      * Enum variants may be structs
      * Destructors can be added to all nominal types with the Drop trait
      * Structs and nullary enum variants may be constants
      * Values that cannot be implicitly copied are now automatically moved
        without writing `move` explicitly
      * `&T` may now be coerced to `*T`
      * Coercions happen in `let` statements as well as function calls
      * `use` statements now take crate-relative paths
      * The module and type namespaces have been merged so that static
        method names can be resolved under the trait in which they are
        declared

   * Improved support for language features
      * Trait inheritance works in many scenarios
      * More support for explicit self arguments in methods - `self`, `&self`
        `@self`, and `~self` all generally work as expected
      * Static methods work in more situations
      * Experimental: Traits may declare default methods for the implementations
        to use

   * Libraries
      * New condition handling system in `core::condition`
      * Timsort added to `std::sort`
      * New priority queue, `std::priority_queue`
      * Pipes for serializable types, `std::flatpipes'
      * Serialization overhauled to be trait-based
      * Expanded `getopts` definitions
      * Moved futures to `std`
      * More functions are pure now
      * `core::comm` renamed to `oldcomm`. Still deprecated
      * `rustdoc` and `cargo` are libraries now

   * Misc
      * Added a preliminary REPL, `rusti`
      * License changed from MIT to dual MIT/APL2

Version 0.4 (October 2012)
--------------------------

   * ~2000 changes, numerous bugfixes

   * Syntax
      * All keywords are now strict and may not be used as identifiers anywhere
      * Keyword removal: 'again', 'import', 'check', 'new', 'owned', 'send',
        'of', 'with', 'to', 'class'.
      * Classes are replaced with simpler structs
      * Explicit method self types
      * `ret` became `return` and `alt` became `match`
      * `import` is now `use`; `use is now `extern mod`
      * `extern mod { ... }` is now `extern { ... }`
      * `use mod` is the recommended way to import modules
      * `pub` and `priv` replace deprecated export lists
      * The syntax of `match` pattern arms now uses fat arrow (=>)
      * `main` no longer accepts an args vector; use `os::args` instead

   * Semantics
      * Trait implementations are now coherent, ala Haskell typeclasses
      * Trait methods may be static
      * Argument modes are deprecated
      * Borrowed pointers are much more mature and recommended for use
      * Strings and vectors in the static region are stored in constant memory
      * Typestate was removed
      * Resolution rewritten to be more reliable
      * Support for 'dual-mode' data structures (freezing and thawing)

   * Libraries
      * Most binary operators can now be overloaded via the traits in
        `core::ops'
      * `std::net::url` for representing URLs
      * Sendable hash maps in `core::send_map`
      * `core::task' gained a (currently unsafe) task-local storage API

   * Concurrency
      * An efficient new intertask communication primitive called the pipe,
        along with a number of higher-level channel types, in `core::pipes`
      * `std::arc`, an atomically reference counted, immutable, shared memory
        type
      * `std::sync`, various exotic synchronization tools based on arcs and pipes
      * Futures are now based on pipes and sendable
      * More robust linked task failure
      * Improved task builder API

   * Other
      * Improved error reporting
      * Preliminary JIT support
      * Preliminary work on precise GC
      * Extensive architectural improvements to rustc
      * Begun a transition away from buggy C++-based reflection (shape) code to
        Rust-based (visitor) code
      * All hash functions and tables converted to secure, randomized SipHash

Version 0.3  (July 2012)
------------------------

   * ~1900 changes, numerous bugfixes

   * New coding conveniences
      * Integer-literal suffix inference
      * Per-item control over warnings, errors
      * #[cfg(windows)] and #[cfg(unix)] attributes
      * Documentation comments
      * More compact closure syntax
      * 'do' expressions for treating higher-order functions as
        control structures
      * *-patterns (wildcard extended to all constructor fields)

   * Semantic cleanup
      * Name resolution pass and exhaustiveness checker rewritten
      * Region pointers and borrow checking supersede alias
        analysis
      * Init-ness checking is now provided by a region-based liveness
        pass instead of the typestate pass; same for last-use analysis
      * Extensive work on region pointers

   * Experimental new language features
      * Slices and fixed-size, interior-allocated vectors
      * #!-comments for lang versioning, shell execution
      * Destructors and iface implementation for classes;
        type-parameterized classes and class methods
      * 'const' type kind for types that can be used to implement
        shared-memory concurrency patterns

   * Type reflection

   * Removal of various obsolete features
      * Keywords: 'be', 'prove', 'syntax', 'note', 'mutable', 'bind',
                 'crust', 'native' (now 'extern'), 'cont' (now 'again')

      * Constructs: do-while loops ('do' repurposed), fn binding,
                    resources (replaced by destructors)

   * Compiler reorganization
      * Syntax-layer of compiler split into separate crate
      * Clang (from LLVM project) integrated into build
      * Typechecker split into sub-modules

   * New library code
      * New time functions
      * Extension methods for many built-in types
      * Arc: atomic-refcount read-only / exclusive-use shared cells
      * Par: parallel map and search routines
      * Extensive work on libuv interface
      * Much vector code moved to libraries
      * Syntax extensions: #line, #col, #file, #mod, #stringify,
        #include, #include_str, #include_bin

   * Tool improvements
      * Cargo automatically resolves dependencies

Version 0.2  (March 2012)
-------------------------

   * >1500 changes, numerous bugfixes

   * New docs and doc tooling

   * New port: FreeBSD x86_64

   * Compilation model enhancements
      * Generics now specialized, multiply instantiated
      * Functions now inlined across separate crates

   * Scheduling, stack and threading fixes
      * Noticeably improved message-passing performance
      * Explicit schedulers
      * Callbacks from C
      * Helgrind clean

   * Experimental new language features
      * Operator overloading
      * Region pointers
      * Classes

   * Various language extensions
      * C-callback function types: 'crust fn ...'
      * Infinite-loop construct: 'loop { ... }'
      * Shorten 'mutable' to 'mut'
      * Required mutable-local qualifier: 'let mut ...'
      * Basic glob-exporting: 'export foo::*;'
      * Alt now exhaustive, 'alt check' for runtime-checked
      * Block-function form of 'for' loop, with 'break' and 'ret'.

   * New library code
      * AST quasi-quote syntax extension
      * Revived libuv interface
      * New modules: core::{future, iter}, std::arena
      * Merged per-platform std::{os*, fs*} to core::{libc, os}
      * Extensive cleanup, regularization in libstd, libcore

Version 0.1  (January 20, 2012)
-------------------------------

   * Most language features work, including:
      * Unique pointers, unique closures, move semantics
      * Interface-constrained generics
      * Static interface dispatch
      * Stack growth
      * Multithread task scheduling
      * Typestate predicates
      * Failure unwinding, destructors
      * Pattern matching and destructuring assignment
      * Lightweight block-lambda syntax
      * Preliminary macro-by-example

   * Compiler works with the following configurations:
      * Linux: x86 and x86_64 hosts and targets
      * MacOS: x86 and x86_64 hosts and targets
      * Windows: x86 hosts and targets

   * Cross compilation / multi-target configuration supported.

   * Preliminary API-documentation and package-management tools included.

Known issues:

   * Documentation is incomplete.

   * Performance is below intended target.

   * Standard library APIs are subject to extensive change, reorganization.

   * Language-level versioning is not yet operational - future code will
     break unexpectedly.
