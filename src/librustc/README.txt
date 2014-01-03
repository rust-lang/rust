An informal guide to reading and working on the rustc compiler.
==================================================================

If you wish to expand on this document, or have a more experienced
Rust contributor add anything else to it, please get in touch:

https://github.com/mozilla/rust/wiki/Note-development-policy
("Communication" subheading)

or file a bug:

https://github.com/mozilla/rust/issues

Your concerns are probably the same as someone else's.


High-level concepts
===================

Rustc consists of the following subdirectories:

front/    - front-end: attributes, conditional compilation
middle/   - middle-end: name resolution, typechecking, LLVM code
                  generation
back/     - back-end: linking and ABI
metadata/ - encoder and decoder for data required by
                    separate compilation
driver/   - command-line processing, main() entrypoint
util/     - ubiquitous types and helper functions
lib/      - bindings to LLVM

The files concerned purely with syntax -- that is, the AST, parser,
pretty-printer, lexer, macro expander, and utilities for traversing
ASTs -- are in a separate crate called "syntax", whose files are in
./../libsyntax, where . is the current directory (that is, the parent
directory of front/, middle/, back/, and so on).

The entry-point for the compiler is main() in lib.rs, and
this file sequences the various parts together.


The 3 central data structures:
------------------------------

#1: ./../libsyntax/ast.rs defines the AST. The AST is treated as immutable
    after parsing, but it depends on mutable context data structures
    (mainly hash maps) to give it meaning.

      - Many -- though not all -- nodes within this data structure are
        wrapped in the type `spanned<T>`, meaning that the front-end has
        marked the input coordinates of that node. The member .node is
        the data itself, the member .span is the input location (file,
        line, column; both low and high).

      - Many other nodes within this data structure carry a
        def_id. These nodes represent the 'target' of some name
        reference elsewhere in the tree. When the AST is resolved, by
        middle/resolve.rs, all names wind up acquiring a def that they
        point to. So anything that can be pointed-to by a name winds
        up with a def_id.

#2: middle/ty.rs defines the datatype sty.  This is the type that
    represents types after they have been resolved and normalized by
    the middle-end. The typeck phase converts every ast type to a
    ty::sty, and the latter is used to drive later phases of
    compilation.  Most variants in the ast::ty tag have a
    corresponding variant in the ty::sty tag.

#3: lib/llvm.rs defines the exported types ValueRef, TypeRef,
    BasicBlockRef, and several others. Each of these is an opaque
    pointer to an LLVM type, manipulated through the lib::llvm
    interface.


Control and information flow within the compiler:
-------------------------------------------------

- main() in lib.rs assumes control on startup. Options are
  parsed, platform is detected, etc.

- ./../libsyntax/parse/parser.rs parses the input files and produces an AST
  that represents the input crate.

- Multiple middle-end passes (middle/resolve.rs, middle/typeck.rs)
  analyze the semantics of the resulting AST. Each pass generates new
  information about the AST and stores it in various environment data
  structures. The driver passes environments to each compiler pass
  that needs to refer to them.

- Finally middle/trans.rs translates the Rust AST to LLVM bitcode in a
  type-directed way. When it's finished synthesizing LLVM values,
  rustc asks LLVM to write them out in some form (.bc, .o) and
  possibly run the system linker.
