# LLDB Internals

LLDB's debug info processing relies on a set of extensible interfaces largely defined in
[lldb/src/Plugins][lldb_plugins]. These are meant to allow third-party compiler developers to add
language support that is loaded at run-time by LLDB, but at time of writing (Nov 2025) the public
API has not been settled on, so plugins exist either in LLDB itself or in standalone forks of LLDB.

[lldb_plugins]: https://github.com/llvm/llvm-project/tree/main/lldb/source/Plugins

Typically, language support will be written as a pipeline of these plugins: `*ASTParser` ->
`TypeSystem` -> `ExpressionParser`/`Language`.

Here are some existing implementations of LLDB's plugin API:

* [Apple's fork with support for Swift](https://github.com/swiftlang/llvm-project)
* [CodeLLDB's former fork with support for Rust](https://archive.softwareheritage.org/browse/origin/directory/?branch=refs/heads/codelldb/16.x&origin_url=https://github.com/vadimcn/llvm-project&path=lldb/source/Plugins/TypeSystem/Rust&timestamp=2023-09-11T04:55:10Z)
* [A work in progress reimplementation of Rust support](https://github.com/Walnut356/llvm-project/tree/lldbrust/19.x)
* [A Rust expression parser plugin](https://github.com/tromey/lldb/tree/a0fc10ce0dacb3038b7302fff9f6cb8cb34b37c6/source/Plugins/ExpressionParser/Rust).
This was written before the `TypeSystem` API was created. Due to the freeform nature of expression parsing, the
underlyng lexing, parsing, function calling, etc. should still offer valuable insights.

## Rust Support and TypeSystemClang

As mentioned in the debug info overview, LLDB has partial Rust support. To further clarify, Rust
uses the plugin-pipeline that was built for C/C++ (though it contains some helpers for Rust enum
types), which relies directly on the `clang` compiler's representation of types. This imposes heavy
restrictions on how much we can change when LLDB's output doesn't match what we want. Some
workarounds can help, but at the end of the day Rust's needs are secondary compared to making sure
C and C++ compilation and debugging work correctly.

LLDB is receptive to adding a `TypeSystemRust`, but it is a massive undertaking. This section serves
to not only document how we currently interact with [`TypeSystemClang`][ts_clang], but also as light
guidance on implementing a `TypeSystemRust` in the future.

[ts_clang]: https://github.com/llvm/llvm-project/tree/main/lldb/source/Plugins/TypeSystem/Clang

It is worth noting that a `TypeSystem` directly interacting with the target language's compiler is
the intention, but it is not a requirement. One can create all the necessary supporting types within
their plugin implementation.

> Note: LLDB's documentation, including comments in the source code, is pretty sparse. Trying to
> understand how language support works by reading `TypeSystemClang`'s implementation is somewhat
> difficult due to the added requirement of understanding the `clang` compiler's internals. It is
> recommended to look at the 2 `TypeSystemRust` implementations listed above, as they are written
> "from scratch" without leveraging a compiler's type representation. They are relatively close to
> the minimum necessary to implement language support.

## DWARF vs PDB

LLDB is unique in being able to handle both DWARF and PDB debug information. This does come with
some added complexity. To complicate matters further, PDB support is split between `dia`, which
relies on the `msdia140.dll` library distributed with Visual Studio, and `native`, which is written
from scratch using publicly available information about the PDB format.

> Note: `dia` was the default up to LLDB version 21. `native` is the new default as of
> LLDB 22's release. There are plans to deprecate and completely remove the `dia`-based plugins. As
> such, only `native` parsing will be discussed below. For progress, please see
> [this discourse thread][dia_discourse] and the relevant [tracking issue][dia_tracking].
>
> `native` can be toggled via the `plugin.symbol-file.pdb.reader` setting added in LLDB 22 or using
> the environment variable `LLDB_USE_NATIVE_PDB_READER=0/1`

[dia_discourse]: https://discourse.llvm.org/t/rfc-removing-the-dia-pdb-plugin-from-lldb/87827
[dia_tracking]: https://github.com/llvm/llvm-project/issues/114906

## Debug Node Parsing

The first step is to process the raw debug nodes into something usable. This primarily occurs in
the [`DWARFASTParser`][dwarf_ast] and [`PdbAstBuilder`][pdb_ast] classes. These classes are fed a
deserialized form of the debug info generated from [`SymbolFileDWARF`][sf_dwarf] and
[`SymbolFileNativePDB`][sf_pdb] respectively. The `SymbolFile` implementers make almost no
transformations to the underlying debug info before passing it to the parsers. For both PDB and
DWARF, the debug info is read using LLVM's debug info handlers.

[dwarf_ast]: https://github.com/llvm/llvm-project/tree/main/lldb/source/Plugins/SymbolFile/DWARF
[pdb_ast]: https://github.com/llvm/llvm-project/tree/main/lldb/source/Plugins/SymbolFile/NativePDB
[sf_dwarf]: https://github.com/llvm/llvm-project/blob/main/lldb/source/Plugins/SymbolFile/DWARF/SymbolFileDWARF.h
[sf_pdb]: https://github.com/llvm/llvm-project/blob/main/lldb/source/Plugins/SymbolFile/NativePDB/SymbolFileNativePDB.h

The parsers translate the nodes into more convenient formats for LLDB's purposes. For `clang`, these
formats are `clang::QualType`, `clang::Decl`, and `clang::DeclContext`, which are the types `clang`
uses internally when compiling C and C++. Again, using the compiler's representation of types is not a
requirement, but the plugin system was built with it as a possibility.

> Note: The above types will be referred to language-agnostically as `LangType`, `Decl`, and
`DeclContext` when the specific implementation details of `TypeSystemClang` are not relevant.

`LangType` represents a type. This includes information such as the name of the type, the size and
alignment, its classification (e.g. struct, primitive, pointer), its qualifiers (e.g.
`const`, `volatile`), template arguments, function argument and return types, etc. [Here][rust_type]
is an example of what a `RustType` might look like.

[rust_type]: https://github.com/Walnut356/llvm-project/blob/13bcfd502452606d69faeea76aec3a06db554af9/lldb/source/Plugins/TypeSystem/Rust/TypeSystemRust.h#L618

`Decl` represents any kind of declaration. It could be a type, a variable, a static field of a
struct, the value that a static or const is initialized with, etc.

`DeclContext` more or less represents a scope. `DeclContext`s typically contain `Decl`s and other
`DeclContexts`, though the relationship isn't that straight forward. For example, a function can be
both a `Decl` (because function signatures are types), **and** a `DeclContext` (because functions
contain variable declarations, nested functions declarations, etc.).

The translation process can be quite verbose, but is usually straightforward. Much of the work here
is dependant on the exact information needed to fill out `LangType`, `Decl`, and `DeclContext`.

Once a node is translated, a pointer to it is type-erased (`void*`) and wrapped in `CompilerType`,
`CompilerDecl`, or `CompilerDeclContext`. These wrappers associate the them with the `TypeSystem`
that owns them. Methods on these objects delegates to the `TypeSystem`, which casts the `void*` back
to the appropriate `LangType*`/`Decl*`/`DeclContext*` and operates on the internals. In Rust terms,
the relationship looks something like this:

```Rust
struct CompilerType {
    inner_type: *mut c_void,
    type_system: Arc<dyn TypeSystem>,
}

impl CompilerType {
    pub fn get_byte_size(&self) -> usize {
        self.type_system.get_byte_size(self.lang_type)
    }

}

...

impl TypeSystem for TypeSystemLang {
    pub fn get_byte_size(lang_type: *mut c_void) -> usize {
        let lang_type = lang_type as *mut LangType;

        // Operate on the internals of the LangType to
        // determine its size
        ...
    }
}
```

## Type Systems

The [`TypeSystem` interface][ts_interface] has 3 major purposes:

[ts_interface]: https://github.com/llvm/llvm-project/blob/main/lldb/include/lldb/Symbol/TypeSystem.h#L69

1. Act as the "sole authority" of a language's types. This allows the type system to be added to
LLDB's "pool" of type systems. When an executable is loaded, the target language is determined, and
the pool is queried to find a `TypeSystem` that claims it can handle the language. One can also use
the `TypeSystem` to retrieve the backing `SymbolFile`, search for types, and synthesize basic types
that might not exist in the debug info (e.g. primitives, arrays-of-`T`, pointers-to-`T`).
2. Manage the lifetimes of the `LangType`, `Decl`, and `DeclContext` objects
3. Customize the "defaults" of how those types appear and how they can be interacted with.

The first two functions are pretty straightforward so we will focus on the third.

Many of the functions in the `TypeSystem` interface will look familiar if you have worked with the
visualizer scripts. These functions underpin `SBType` the `SBValue` functions with matching names.
For example, `TypeSystem::GetFormat` returns the default format for the type if no custom formatter
has been applied to it.

Of particular note are `GetIndexOfChildWithName` and `GetNumChildren`. The `TypeSystem` versions of
these functions operate on a *type*, not a value like the `SBValue` versions. The values returned
from the `TypeSystem` functions dictate what parts of the struct can be interacted with *at all* by
the rest of LLDB. If a field is ommitted, that field effectively no longer exists to LLDB.

Additionally, since they do not work with objects, there is no underlying memory to inspect or
interpret. Essentially, this means these functions do not have the same purpose as their equivalent
`SyntheticProvider` functions. There is no way to determine how many elements a `Vec` has or what
address those elements live at. It is also not possible to determine the value of the discriminant
of a sum-type.

Ideally, the `TypeSystem` should expose types as they appear in the debug info with as few
alterations as possible. LLDB's synthetics and frontend can handle making the type pretty. If some
piece of information is useless, the Rust compiler should be altered to not output that debug info
in the first place.

## Expression Parsing

The `TypeSystem` is typically written to have a counterpart that can handle expression parsing. It
requires implementing a few extra functions in the `TypeSystem` interface. The bulk of the
expression parsing code should live in [lldb/source/Plugins/ExpressionParser][expr].

[expr]: https://github.com/llvm/llvm-project/tree/main/lldb/source/Plugins/ExpressionParser

There isn't too much of note about the parser. It requires implementing a simple interpreter that
can handle (possibly simplified) Rust syntax. They operate on `lldb::ValueObject`s, which are the
objects that underpin `SBValue`.

## Language

The [`Language` plugins][lang_plugin] are the C++ equivalent to the Python visualizer scripts. They
operate on `SBValue` objects for the same purpose: creating synthetic children and pretty-printing.
The [CPlusPlusLanguage's implementations][cpp_lang] for the LibCxx types are great resources to
learn how visualizers should be written.

[lang_plugin]: https://github.com/llvm/llvm-project/tree/main/lldb/source/Plugins/Language
[cpp_lang]: https://github.com/llvm/llvm-project/tree/main/lldb/source/Plugins/Language/CPlusPlus

These plugins can access LLDB's private internals (including the underlying `TypeSystem`), so
synthetic/summary providers written as a `Language` plugin can provide higher quality output than
their python equivalent.

While debug node parsing, type systems, and expression parsing are all closely tied to eachother,
the `Language` plugin is encapsulated more and thus can be written "standalone" for any language
that an existing type system supports. Due to the lower barrier of entry, a `RustLanguage` plugin
may be a good stepping stone towards full language support in LLDB.

## Visualizers

WIP