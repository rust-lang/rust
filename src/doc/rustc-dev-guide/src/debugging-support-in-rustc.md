# Debugging support in the Rust compiler

This document explains the state of debugging tools support in the Rust compiler (rustc).
It gives an overview of GDB, LLDB, WinDbg/CDB,
as well as infrastructure around Rust compiler to debug Rust code.
If you want to learn how to debug the Rust compiler itself,
see [Debugging the Compiler].

The material is gathered from the video,
[Tom Tromey discusses debugging support in rustc].

## Preliminaries

### Debuggers

According to Wikipedia

> A [debugger or debugging tool] is a computer program that is used to test and debug
> other programs (the "target" program).

Writing a debugger from scratch for a language requires a lot of work, especially if
debuggers have to be supported on various platforms. GDB and LLDB, however, can be
extended to support debugging a language. This is the path that Rust has chosen.
This document's main goal is to document the said debuggers support in Rust compiler.

### DWARF

According to the [DWARF] standard website

> DWARF is a debugging file format used by many compilers and debuggers to support source level
> debugging. It addresses the requirements of a number of procedural languages,
> such as C, C++, and Fortran, and is designed to be extensible to other languages.
> DWARF is architecture independent and applicable to any processor or operating system.
> It is widely used on Unix, Linux and other operating systems,
> as well as in stand-alone environments.

DWARF reader is a program that consumes the DWARF format and creates debugger compatible output.
This program may live in the compiler itself.  DWARF uses a data structure called
Debugging Information Entry (DIE) which stores the information as "tags" to denote functions,
variables etc., e.g., `DW_TAG_variable`, `DW_TAG_pointer_type`, `DW_TAG_subprogram` etc.
You can also invent your own tags and attributes.

### CodeView/PDB

[PDB] (Program Database) is a file format created by Microsoft that contains debug information.
PDBs can be consumed by debuggers such as WinDbg/CDB and other tools to display debug information.
A PDB contains multiple streams that describe debug information about a specific binary such
as types, symbols, and source files used to compile the given binary. CodeView is another
format which defines the structure of [symbol records] and [type records] that appear within
PDB streams.

## Supported debuggers

### GDB

#### Rust expression parser

To be able to show debug output, we need an expression parser.
This (GDB) expression parser is written in [Bison],
and can parse only a subset of Rust expressions.
GDB parser was written from scratch and has no relation to any other parser,
including that of rustc.

GDB has Rust-like value and type output. It can print values and types in a way
that look like Rust syntax in the output. Or when you print a type as [ptype] in GDB,
it also looks like Rust source code. Checkout the documentation in the [manual for GDB/Rust].

#### Parser extensions

Expression parser has a couple of extensions in it to facilitate features that you cannot do
with Rust. Some limitations are listed in the [manual for GDB/Rust]. There is some special
code in the DWARF reader in GDB to support the extensions.

A couple of examples of DWARF reader support needed are as follows:

1. Enum: Needed for support for enum types.
   The Rust compiler writes the information about enum into DWARF,
   and GDB reads the DWARF to understand where is the tag field,
   or if there is a tag field,
   or if the tag slot is shared with non-zero optimization etc.

2. Dissect trait objects: DWARF extension where the trait object's description in the DWARF
   also points to a stub description of the corresponding vtable which in turn points to the
   concrete type for which this trait object exists. This means that you can do a `print *object`
   for that trait object, and GDB will understand how to find the correct type of the payload in
   the trait object.

**TODO**: Figure out if the following should be mentioned in the GDB-Rust document rather than
this guide page so there is no duplication. This is regarding the following comments:

[This comment by Tom](https://github.com/rust-lang/rustc-dev-guide/pull/316#discussion_r284027340)
> gdb's Rust extensions and limitations are documented in the gdb manual:
https://sourceware.org/gdb/onlinedocs/gdb/Rust.html -- however, this neglects to mention that
gdb convenience variables and registers follow the gdb $ convention, and that the Rust parser
implements the gdb @ extension.

[This question by Aman](https://github.com/rust-lang/rustc-dev-guide/pull/316#discussion_r285401353)
> @tromey do you think we should mention this part in the GDB-Rust document rather than this
document so there is no duplication etc.?

### LLDB

#### Rust expression parser

This expression parser is written in C++. It is a type of [Recursive Descent parser].
It implements slightly less of the Rust language than GDB.
LLDB has Rust-like value and type output.

#### Developer notes

* LLDB has a plugin architecture but that does not work for language support.
* GDB generally works better on Linux.

### WinDbg/CDB

Microsoft provides [Windows Debugging Tools] such as the Windows Debugger (WinDbg) and
the Console Debugger (CDB) which both support debugging programs written in Rust. These
debuggers parse the debug info for a binary from the `PDB`, if available, to construct a
visualization to serve up in the debugger.

#### Natvis

Both WinDbg and CDB support defining and viewing custom visualizations for any given type
within the debugger using the Natvis framework. The Rust compiler defines a set of Natvis
files that define custom visualizations for a subset of types in the standard libraries such
as, `std`, `core`, and `alloc`. These Natvis files are embedded into `PDBs` generated by the
`*-pc-windows-msvc` target triples to automatically enable these custom visualizations when
debugging. This default can be overridden by setting the `strip` rustc flag to either `debuginfo`
or `symbols`.

Rust has support for embedding Natvis files for crates outside of the standard libraries by
using the `#[debugger_visualizer]` attribute.
For more details on how to embed debugger visualizers,
please refer to the section on the [`debugger_visualizer` attribute].

## DWARF and `rustc`

[DWARF] is the standard way compilers generate debugging information that debuggers read.
It is _the_ debugging format on macOS and Linux.
It is a multi-language and extensible format,
and is mostly good enough for Rust's purposes.
Hence, the current implementation reuses DWARF's concepts.
This is true even if some of the concepts in DWARF do not align with Rust semantically because,
generally, there can be some kind of mapping between the two.

We have some DWARF extensions that the Rust compiler emits and the debuggers understand that
are _not_ in the DWARF standard.

* Rust compiler will emit DWARF for a virtual table, and this `vtable` object will have a
  `DW_AT_containing_type` that points to the real type. This lets debuggers dissect a trait object
   pointer to correctly find the payload. E.g., here's such a DIE, from a test case in the gdb
   repository:

   ```asm
   <1><1a9>: Abbrev Number: 3 (DW_TAG_structure_type)
      <1aa>   DW_AT_containing_type: <0x1b4>
      <1ae>   DW_AT_name        : (indirect string, offset: 0x23d): vtable
      <1b2>   DW_AT_byte_size   : 0
      <1b3>   DW_AT_alignment   : 8
   ```

* The other extension is that the Rust compiler can emit a tagless discriminated union.
  See [DWARF feature request] for this item.

### Current limitations of DWARF

* Traits - require a bigger change than normal to DWARF, on how to represent Traits in DWARF.
* DWARF provides no way to differentiate between Structs and Tuples. Rust compiler emits
fields with `__0` and debuggers look for a sequence of such names to overcome this limitation.
For example, in this case the debugger would look at a field via `x.__0` instead of `x.0`.
This is resolved via the Rust parser in the debugger so now you can do `x.0`.

DWARF relies on debuggers to know some information about platform ABI.
Rust does not do that all the time.

## Developer notes

This section is from the talk about certain aspects of development.

## What is missing

### Code signing for LLDB debug server on macOS

According to Wikipedia, [System Integrity Protection] is

> System Integrity Protection (SIP, sometimes referred to as rootless) is a security feature
> of Apple's macOS operating system introduced in OS X El Capitan. It comprises a number of
> mechanisms that are enforced by the kernel. A centerpiece is the protection of system-owned
> files and directories against modifications by processes without a specific "entitlement",
> even when executed by the root user or a user with root privileges (sudo).

It prevents processes using `ptrace` syscall. If a process wants to use `ptrace` it has to be
code signed. The certificate that signs it has to be trusted on your machine.

See [Apple developer documentation for System Integrity Protection].

We may need to sign up with Apple and get the keys to do this signing. Tom has looked into if
Mozilla cannot do this because it is at the maximum number of
keys it is allowed to sign. Tom does not know if Mozilla could get more keys.

Alternatively, Tom suggests that maybe a Rust legal entity is needed to get the keys via Apple.
This problem is not technical in nature. If we had such a key we could sign GDB as well and
ship that.

### DWARF and Traits

Rust traits are not emitted into DWARF at all. The impact of this is calling a method `x.method()`
does not work as is. The reason being that method is implemented by a trait, as opposed
to a type. That information is not present so finding trait methods is missing.

DWARF has a notion of interface types (possibly added for Java). Tom's idea was to use this
interface type as traits.

DWARF only deals with concrete names, not the reference types. So, a given implementation of a
trait for a type would be one of these interfaces (`DW_tag_interface` type). Also, the type for
which it is implemented would describe all the interfaces this type implements. This requires a
DWARF extension.

Issue on Github: [https://github.com/rust-lang/rust/issues/33014]

## Typical process for a Debug Info change (LLVM)

LLVM has Debug Info (DI) builders. This is the primary thing that Rust calls into.
This is why we need to change LLVM first because that is emitted first and not DWARF directly.
This is a kind of metadata that you construct and hand-off to LLVM. For the Rustc/LLVM hand-off
some LLVM DI builder methods are called to construct representation of a type.

The steps of this process are as follows:

1. LLVM needs changing.

   LLVM does not emit Interface types at all, so this needs to be implemented in the LLVM first.

   Get sign off on LLVM maintainers that this is a good idea.

2. Change the DWARF extension.

3. Update the debuggers.

   Update DWARF readers, expression evaluators.

4. Update Rust compiler.

   Change it to emit this new information.

### Procedural macro stepping

A deeply profound question is that how do you actually debug a procedural macro?
What is the location you emit for a macro expansion? Consider some of the following cases -

* You can emit location of the invocation of the macro.
* You can emit the location of the definition of the macro.
* You can emit locations of the content of the macro.

RFC: [https://github.com/rust-lang/rfcs/pull/2117]

Focus is to let macros decide what to do. This can be achieved by having some kind of attribute
that lets the macro tell the compiler where the line marker should be. This affects where you
set the breakpoints and what happens when you step it.

## Source file checksums in debug info

Both DWARF and CodeView (PDB) support embedding a cryptographic hash of each source file that
contributed to the associated binary.

The cryptographic hash can be used by a debugger to verify that the source file matches the
executable. If the source file does not match, the debugger can provide a warning to the user.

The hash can also be used to prove that a given source file has not been modified since it was
used to compile an executable. Because MD5 and SHA1 both have demonstrated vulnerabilities,
using SHA256 is recommended for this application.

The Rust compiler stores the hash for each source file in the corresponding `SourceFile` in
the `SourceMap`. The hashes of input files to external crates are stored in `rlib` metadata.

A default hashing algorithm is set in the target specification. This allows the target to
specify the best hash available, since not all targets support all hash algorithms.

The hashing algorithm for a target can also be overridden with the `-Z source-file-checksum=`
command-line option.

#### DWARF 5
DWARF version 5 supports embedding an MD5 hash to validate the source file version in use.
DWARF 5 - Section 6.2.4.1 opcode DW_LNCT_MD5

#### LLVM
LLVM IR supports MD5 and SHA1 (and SHA256 in LLVM 11+) source file checksums in the DIFile node.

[LLVM DIFile documentation](https://llvm.org/docs/LangRef.html#difile)

#### Microsoft Visual C++ Compiler /ZH option
The MSVC compiler supports embedding MD5, SHA1, or SHA256 hashes in the PDB using the `/ZH`
compiler option.

[MSVC /ZH documentation](https://docs.microsoft.com/en-us/cpp/build/reference/zh)

#### Clang
Clang always embeds an MD5 checksum, though this does not appear in documentation.

## Future work

#### Name mangling changes

* New demangler in `libiberty` (gcc source tree).
* New demangler in LLVM or LLDB.

**TODO**: Check the location of the demangler source. [#1157](https://github.com/rust-lang/rustc-dev-guide/issues/1157)

#### Reuse Rust compiler for expressions

This is an important idea because debuggers by and large do not try to implement type
inference. You need to be much more explicit when you type into the debugger than your
actual source code. So, you cannot just copy and paste an expression from your source
code to debugger and expect the same answer but this would be nice. This can be helped
by using compiler.

It is certainly doable but it is a large project. You certainly need a bridge to the
debugger because the debugger alone has access to the memory. Both GDB (gcc) and LLDB (clang)
have this feature. LLDB uses Clang to compile code to JIT and GDB can do the same with GCC.

Both debuggers expression evaluation implement both a superset and a subset of Rust.
They implement just the expression language,
but they also add some extensions like GDB has convenience variables.
Therefore, if you are taking this route,
then you not only need to do this bridge,
but may have to add some mode to let the compiler understand some extensions.

[Tom Tromey discusses debugging support in rustc]: https://www.youtube.com/watch?v=elBxMRSNYr4
[Debugging the Compiler]: compiler-debugging.md
[debugger or debugging tool]: https://en.wikipedia.org/wiki/Debugger
[Bison]: https://www.gnu.org/software/bison/
[ptype]: https://ftp.gnu.org/old-gnu/Manuals/gdb/html_node/gdb_109.html
[rust-lang/lldb wiki page]: https://github.com/rust-lang/lldb/wiki
[DWARF]: http://dwarfstd.org
[manual for GDB/Rust]: https://sourceware.org/gdb/onlinedocs/gdb/Rust.html
[GDB Bugzilla]: https://sourceware.org/bugzilla/
[Recursive Descent parser]: https://en.wikipedia.org/wiki/Recursive_descent_parser
[System Integrity Protection]: https://en.wikipedia.org/wiki/System_Integrity_Protection
[https://github.com/rust-dev-tools/gdb]: https://github.com/rust-dev-tools/gdb
[DWARF feature request]: http://dwarfstd.org/ShowIssue.php?issue=180517.2
[https://docs.python.org/3/c-api/stable.html]: https://docs.python.org/3/c-api/stable.html
[https://github.com/rust-lang/rfcs/pull/2117]: https://github.com/rust-lang/rfcs/pull/2117
[https://github.com/rust-lang/rust/issues/33014]: https://github.com/rust-lang/rust/issues/33014
[https://github.com/rust-lang/rust/issues/34457]: https://github.com/rust-lang/rust/issues/34457
[Apple developer documentation for System Integrity Protection]: https://developer.apple.com/library/archive/releasenotes/MacOSX/WhatsNewInOSX/Articles/MacOSX10_11.html#//apple_ref/doc/uid/TP40016227-SW11
[https://github.com/rust-lang/lldb]: https://github.com/rust-lang/lldb
[https://github.com/rust-lang/llvm-project]: https://github.com/rust-lang/llvm-project
[PDB]: https://llvm.org/docs/PDB/index.html
[symbol records]: https://llvm.org/docs/PDB/CodeViewSymbols.html
[type records]: https://llvm.org/docs/PDB/CodeViewTypes.html
[Windows Debugging Tools]: https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/
[`debugger_visualizer` attribute]: https://doc.rust-lang.org/nightly/reference/attributes/debugger.html#the-debugger_visualizer-attribute
