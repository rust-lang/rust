// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Debug Info Module
//!
//! This module serves the purpose of generating debug symbols. We use LLVM's
//! [source level debugging](http://!llvm.org/docs/SourceLevelDebugging.html)
//! features for generating the debug information. The general principle is
//! this:
//!
//! Given the right metadata in the LLVM IR, the LLVM code generator is able to
//! create DWARF debug symbols for the given code. The
//! [metadata](http://!llvm.org/docs/LangRef.html#metadata-type) is structured
//! much like DWARF *debugging information entries* (DIE), representing type
//! information such as datatype layout, function signatures, block layout,
//! variable location and scope information, etc. It is the purpose of this
//! module to generate correct metadata and insert it into the LLVM IR.
//!
//! As the exact format of metadata trees may change between different LLVM
//! versions, we now use LLVM
//! [DIBuilder](http://!llvm.org/docs/doxygen/html/classllvm_1_1DIBuilder.html)
//! to create metadata where possible. This will hopefully ease the adaption of
//! this module to future LLVM versions.
//!
//! The public API of the module is a set of functions that will insert the
//! correct metadata into the LLVM IR when called with the right parameters.
//! The module is thus driven from an outside client with functions like
//! `debuginfo::create_local_var_metadata(bcx: block, local: &ast::local)`.
//!
//! Internally the module will try to reuse already created metadata by
//! utilizing a cache. The way to get a shared metadata node when needed is
//! thus to just call the corresponding function in this module:
//!
//!     let file_metadata = file_metadata(crate_context, path);
//!
//! The function will take care of probing the cache for an existing node for
//! that exact file path.
//!
//! All private state used by the module is stored within either the
//! CrateDebugContext struct (owned by the CrateContext) or the
//! FunctionDebugContext (owned by the FunctionContext).
//!
//! This file consists of three conceptual sections:
//! 1. The public interface of the module
//! 2. Module-internal metadata creation functions
//! 3. Minor utility functions
//!
//!
//! ## Recursive Types
//!
//! Some kinds of types, such as structs and enums can be recursive. That means
//! that the type definition of some type X refers to some other type which in
//! turn (transitively) refers to X. This introduces cycles into the type
//! referral graph. A naive algorithm doing an on-demand, depth-first traversal
//! of this graph when describing types, can get trapped in an endless loop
//! when it reaches such a cycle.
//!
//! For example, the following simple type for a singly-linked list...
//!
//! ```
//! struct List {
//!     value: isize,
//!     tail: Option<Box<List>>,
//! }
//! ```
//!
//! will generate the following callstack with a naive DFS algorithm:
//!
//! ```
//! describe(t = List)
//!   describe(t = int)
//!   describe(t = Option<Box<List>>)
//!     describe(t = Box<List>)
//!       describe(t = List) // at the beginning again...
//!       ...
//! ```
//!
//! To break cycles like these, we use "forward declarations". That is, when
//! the algorithm encounters a possibly recursive type (any struct or enum), it
//! immediately creates a type description node and inserts it into the cache
//! *before* describing the members of the type. This type description is just
//! a stub (as type members are not described and added to it yet) but it
//! allows the algorithm to already refer to the type. After the stub is
//! inserted into the cache, the algorithm continues as before. If it now
//! encounters a recursive reference, it will hit the cache and does not try to
//! describe the type anew.
//!
//! This behaviour is encapsulated in the 'RecursiveTypeDescription' enum,
//! which represents a kind of continuation, storing all state needed to
//! continue traversal at the type members after the type has been registered
//! with the cache. (This implementation approach might be a tad over-
//! engineered and may change in the future)
//!
//!
//! ## Source Locations and Line Information
//!
//! In addition to data type descriptions the debugging information must also
//! allow to map machine code locations back to source code locations in order
//! to be useful. This functionality is also handled in this module. The
//! following functions allow to control source mappings:
//!
//! + set_source_location()
//! + clear_source_location()
//! + start_emitting_source_locations()
//!
//! `set_source_location()` allows to set the current source location. All IR
//! instructions created after a call to this function will be linked to the
//! given source location, until another location is specified with
//! `set_source_location()` or the source location is cleared with
//! `clear_source_location()`. In the later case, subsequent IR instruction
//! will not be linked to any source location. As you can see, this is a
//! stateful API (mimicking the one in LLVM), so be careful with source
//! locations set by previous calls. It's probably best to not rely on any
//! specific state being present at a given point in code.
//!
//! One topic that deserves some extra attention is *function prologues*. At
//! the beginning of a function's machine code there are typically a few
//! instructions for loading argument values into allocas and checking if
//! there's enough stack space for the function to execute. This *prologue* is
//! not visible in the source code and LLVM puts a special PROLOGUE END marker
//! into the line table at the first non-prologue instruction of the function.
//! In order to find out where the prologue ends, LLVM looks for the first
//! instruction in the function body that is linked to a source location. So,
//! when generating prologue instructions we have to make sure that we don't
//! emit source location information until the 'real' function body begins. For
//! this reason, source location emission is disabled by default for any new
//! function being translated and is only activated after a call to the third
//! function from the list above, `start_emitting_source_locations()`. This
//! function should be called right before regularly starting to translate the
//! top-level block of the given function.
//!
//! There is one exception to the above rule: `llvm.dbg.declare` instruction
//! must be linked to the source location of the variable being declared. For
//! function parameters these `llvm.dbg.declare` instructions typically occur
//! in the middle of the prologue, however, they are ignored by LLVM's prologue
//! detection. The `create_argument_metadata()` and related functions take care
//! of linking the `llvm.dbg.declare` instructions to the correct source
//! locations even while source location emission is still disabled, so there
//! is no need to do anything special with source location handling here.
//!
//! ## Unique Type Identification
//!
//! In order for link-time optimization to work properly, LLVM needs a unique
//! type identifier that tells it across compilation units which types are the
//! same as others. This type identifier is created by
//! TypeMap::get_unique_type_id_of_type() using the following algorithm:
//!
//! (1) Primitive types have their name as ID
//! (2) Structs, enums and traits have a multipart identifier
//!
//!     (1) The first part is the SVH (strict version hash) of the crate they
//!          wereoriginally defined in
//!
//!     (2) The second part is the ast::NodeId of the definition in their
//!          originalcrate
//!
//!     (3) The final part is a concatenation of the type IDs of their concrete
//!          typearguments if they are generic types.
//!
//! (3) Tuple-, pointer and function types are structurally identified, which
//!     means that they are equivalent if their component types are equivalent
//!     (i.e. (int, int) is the same regardless in which crate it is used).
//!
//! This algorithm also provides a stable ID for types that are defined in one
//! crate but instantiated from metadata within another crate. We just have to
//! take care to always map crate and node IDs back to the original crate
//! context.
//!
//! As a side-effect these unique type IDs also help to solve a problem arising
//! from lifetime parameters. Since lifetime parameters are completely omitted
//! in debuginfo, more than one `Ty` instance may map to the same debuginfo
//! type metadata, that is, some struct `Struct<'a>` may have N instantiations
//! with different concrete substitutions for `'a`, and thus there will be N
//! `Ty` instances for the type `Struct<'a>` even though it is not generic
//! otherwise. Unfortunately this means that we cannot use `ty::type_id()` as
//! cheap identifier for type metadata---we have done this in the past, but it
//! led to unnecessary metadata duplication in the best case and LLVM
//! assertions in the worst. However, the unique type ID as described above
//! *can* be used as identifier. Since it is comparatively expensive to
//! construct, though, `ty::type_id()` is still used additionally as an
//! optimization for cases where the exact same type has been seen before
//! (which is most of the time).
