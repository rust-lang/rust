# UI Test Suite Categories

This is a high-level summary of the organization of the UI test suite (`tests/ui/`). It is not intended to be *prescriptive*, but instead provide a quick survey of existing groupings.

For now, only immediate subdirectories under `tests/ui/` are described, but these subdirectories can themselves include a `SUMMARY.md` to further describe their own organization and intent, should that be helpful.

## `tests/ui/abi`

These tests deal with *Application Binary Interfaces* (ABI), mostly relating to function name mangling (and the `#[no_mangle]` attribute), calling conventions, or compiler flags which affect ABI.

## `tests/ui/allocator`

These tests exercise `#![feature(allocator_api)]` and the `#[global_allocator]` attribute.

See [Allocator traits and `std::heap` #32838](https://github.com/rust-lang/rust/issues/32838).

## `tests/ui/alloc-error`

These tests exercise alloc error handling.

See <https://doc.rust-lang.org/std/alloc/fn.handle_alloc_error.html>.

## `tests/ui/annotate-snippet`

These tests exercise the [`annotate-snippets`]-based emitter implementation.

[`annotate-snippets`] is an initiative to share the diagnostics emitting infrastructure between rustc and cargo to reduce duplicate maintenance effort and divergence. See <https://github.com/rust-lang/rust/issues/59346> about the initiative.

[`annotate-snippets`]: https://github.com/rust-lang/annotate-snippets-rs

## `tests/ui/anon-params`

These tests deal with anonymous parameters (no name, only type), a deprecated feature that becomes a hard error in Edition 2018.

## `tests/ui/argfile`: External files providing command line arguments

These tests exercise rustc reading command line arguments from an externally provided argfile (`@argsfile`).

See [Implement `@argsfile` to read arguments from command line #63576](https://github.com/rust-lang/rust/issues/63576).

## `tests/ui/array-slice-vec`: Arrays, slices and vectors

Exercises various aspects surrounding basic collection types `[]`, `&[]` and `Vec`. E.g. type-checking, out-of-bounds indices, attempted instructions which are allowed in other programming languages, and more.

## `tests/ui/argument-suggestions`: Argument suggestions

Calling a function with the wrong number of arguments causes a compilation failure, but the compiler is able to, in some cases, provide suggestions on how to fix the error, such as which arguments to add or delete. These tests exercise the quality of such diagnostics.

## `tests/ui/asm`: `asm!` macro

These tests exercise the `asm!` macro, which is used for adding inline assembly.

See:

- [Inline assembly | Reference](https://doc.rust-lang.org/reference/inline-assembly.html)
- [`core::arch::asm`](https://doc.rust-lang.org/core/arch/macro.asm.html)
- [`core::arch::global_asm`](https://doc.rust-lang.org/core/arch/macro.global_asm.html)

This directory contains subdirectories representing various architectures such as `riscv` or `aarch64`. If a test is specifically related to an architecture's particularities, it should be placed within the appropriate subdirectory.Architecture-agnostic tests should be placed below `tests/ui/asm/` directly.

## `tests/ui/associated-consts`: Associated Constants

These tests exercise associated constants in traits and impls, on aspects such as definitions, usage, and type checking in associated contexts.

## `tests/ui/associated-inherent-types`: Inherent Associated Types

These tests cover associated types defined directly within inherent impls (not in traits).

See [RFC 0195 Associated items - Inherent associated items](https://github.com/rust-lang/rfcs/blob/master/text/0195-associated-items.md#inherent-associated-items).

## `tests/ui/associated-item`: Associated Items

Tests for all kinds of associated items within traits and implementations. This directory serves as a catch-all for tests that don't fit the other more specific associated item directories.

## `tests/ui/associated-type-bounds`: Associated Type Bounds

These tests exercise associated type bounds, the feature that gives users a shorthand to express nested type bounds that would otherwise need to be expressed with nested `impl Trait` or broken into several `where` clauses.

See:

- [RFC 2289 Associated Type Bounds](https://rust-lang.github.io/rfcs/2289-associated-type-bounds.html)
- [Stabilize associated type bounds (RFC 2289) #122055](https://github.com/rust-lang/rust/pull/122055)

## `tests/ui/associated-types`: Trait Associated Types

Tests focused on associated types. If the associated type is not in a trait definition, it belongs in the `tests/ui/associated-inherent-types/` directory. Aspects exercised include e.g. default associated types, overriding defaults, and type inference.

See [Associated Types | Reference](https://doc.rust-lang.org/reference/items/associated-items.html#associated-types).

## `tests/ui/async-await`: Async/Await

Tests for the async/await related features. E.g. async functions, await expressions, and their interaction with other language features.

## `tests/ui/attributes`: Compiler Attributes

Tests for language attributes and compiler attributes. E.g. built-in attributes like `#[derive(..)]`, `#[cfg(..)]`, and `#[repr(..)]`, or proc-macro attributes. See [Attributes | Reference](https://doc.rust-lang.org/reference/attributes.html).

## `tests/ui/auto-traits`: Auto Traits

There are built-in auto traits (`Send`, `Sync`, etc.) but it is possible to define more with the unstable keyword `auto` through `#![feature(auto_traits)]`.

See [Tracking Issue for auto traits (`auto_traits`) -- formerly called opt-in built-in traits (`optin_builtin_traits`) #13231](https://github.com/rust-lang/rust/issues/13231).

## `tests/ui/autodiff`: Automatic Differentiation

The `#[autodiff]` macro supports automatic differentiation.

See [Tracking Issue for autodiff #124509](https://github.com/rust-lang/rust/issues/124509).

## `tests/ui/autoref-autoderef`: Automatic Referencing/Dereferencing

Tests for automatic referencing and dereferencing behavior, such as automatically adding reference operations (`&` or `&mut`) to make a value match a method's receiver type. Sometimes abbreviated as "auto-ref" or "auto-deref".

## `tests/ui/auxiliary/`: Auxiliary files for tests directly under `tests/ui`.

This top-level `auxiliary` subdirectory contains support files for tests immediately under `tests/ui/`.

**FIXME(#133895)**: tests immediately under `tests/ui/` should be rehomed to more suitable subdirectories, after which this subdirectory can be removed.

## `tests/ui/backtrace/`: Backtraces

Runtime panics and error handling generate backtraces to assist in debugging and diagnostics.

## `tests/ui/bench/`: Benchmarks and performance

This directory was originally meant to contain tests related to time complexity and benchmarking.

However, only a single test was ever added to this category: https://github.com/rust-lang/rust/pull/32062

**FIXME**: It is also unclear what would happen were this test to "fail" - would it cause the test suite to remain stuck on this test for a much greater duration than normal?

## `tests/ui/binding/`: Pattern Binding

Tests for pattern binding in match expressions, let statements, and other binding contexts. E.g. binding modes and refutability. See [Patterns | Reference](https://doc.rust-lang.org/reference/patterns.html).

## `tests/ui/binop/`: Binary operators

Tests for binary operators (such as `==`, `&&` or `^`). E.g. overloading, type checking, and diagnostics for invalid operations.

## `tests/ui/blind/`: `struct` or `mod` inside a `mod` having a duplicate identifier

Tests exercising name resolution.

**FIXME**: Probably move to `tests/ui/resolve/`.

## `tests/ui/block-result/`: Block results and returning

Tests for block expression results. E.g. specifying the correct return types, semicolon handling, type inference, and expression/statement differences (for example, the difference between `1` and `1;`).

## `tests/ui/bootstrap/`: RUSTC_BOOTSTRAP environment variable

Meta tests for stability mechanisms surrounding [`RUSTC_BOOTSTRAP`](https://doc.rust-lang.org/nightly/unstable-book/compiler-environment-variables/RUSTC_BOOTSTRAP.html), which is coordinated between `rustc` and the build system, `bootstrap`.

## `tests/ui/borrowck/`: Borrow Checking

Tests for borrow checking. E.g. lifetime analysis, borrowing rules, and diagnostics.

## `tests/ui/box/`: Box Behavior

Tests for `Box<T>` smart pointer and `#![feature(box_patterns)]`. E.g. allocation, deref coercion, and edge cases in box pattern matching and placement.

See:

- [`std::box::Boxed`](https://doc.rust-lang.org/std/boxed/struct.Box.html)
- [Tracking issue for `box_patterns` feature #29641](https://github.com/rust-lang/rust/issues/29641)

## `tests/ui/btreemap/`: B-Tree Maps

Tests focused on `BTreeMap` collections and their compiler interactions. E.g. collection patterns, iterator behavior, and trait implementations specific to `BTreeMap`. See [`std::collections::BTreeMap`](https://doc.rust-lang.org/std/collections/struct.BTreeMap.html).

## `tests/ui/builtin-superkinds/`: Built-in Trait Hierarchy Tests

Tests for built-in trait hierarchy (Send, Sync, Sized, etc.) and their supertrait relationships. E.g. auto traits and marker trait constraints.

See [RFC 3729: Hierarchy of Sized traits](https://github.com/rust-lang/rfcs/pull/3729).

Defining custom auto traits with the `auto` keyword belongs to `tests/ui/auto-traits/` instead.

## `tests/ui/cast/`: Type Casting

Tests for type casting using the `as` operator. Includes tests for valid/invalid casts between primitive types, trait objects, and custom types. For example, check that trying to cast `i32` into `bool` results in a helpful error message.

See [Type cast expressions | Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html).

## `tests/ui/cfg/`: Configuration Attribute

Tests for `#[cfg]` conditional compilation attribute. E.g. feature flags, target architectures, and other configuration predicates and options.

See [Conditional compilation | Reference](https://doc.rust-lang.org/reference/conditional-compilation.html).

## `tests/ui/check-cfg/`: Configuration Checks

Tests for the `--check-cfg` compiler mechanism  for checking cfg configurations, for `#[cfg(..)]` and `cfg!(..)`.

See [Checking conditional configurations | The rustc book](https://doc.rust-lang.org/rustc/check-cfg.html).

## `tests/ui/closure_context/`: Closure type inference in context

Tests for closure type inference with respect to surrounding scopes, mostly quality of diagnostics.

## `tests/ui/closure-expected-type/`: Closure type inference

Tests targeted at how we deduce the types of closure arguments. This process is a result of some heuristics which take into account the *expected type* we have alongside the *actual types* that we get from inputs.

**FIXME**: Appears to have significant overlap with `tests/ui/closure_context` and `tests/ui/functions-closures/closure-expected-type`. Needs further investigation.

## `tests/ui/closures/`: General Closure Tests

Any closure-focused tests that does not fit in the other more specific closure subdirectories belong here. E.g. syntax, `move`, lifetimes.

## `tests/ui/cmse-nonsecure/`: `C-cmse-nonsecure` ABIs

Tests for `cmse_nonsecure_entry` and `abi_c_cmse_nonsecure_call` ABIs. Used specifically for the Armv8-M architecture, the former marks Secure functions with additional behaviours, such as adding a special symbol and constraining the number of parameters, while the latter alters function pointers to indicate they are non-secure and to handle them differently than usual.

See:

- [`cmse_nonsecure_entry` | The Unstable book](https://doc.rust-lang.org/unstable-book/language-features/cmse-nonsecure-entry.html)
- [`abi_c_cmse_nonsecure_call` | The Unstable book](https://doc.rust-lang.org/beta/unstable-book/language-features/abi-c-cmse-nonsecure-call.html)

## `tests/ui/codegen/`: Code Generation

Tests that exercise code generation. E.g. codegen flags (starting with `-C` on the command line), LLVM IR output, optimizations (and the various `opt-level`s), and target-specific code generation (such as tests specific to `x86_64`).

## `tests/ui/codemap_tests/`: Source Mapping

Tests that exercise source code mapping.

## `tests/ui/coercion/`: Type Coercion

Tests for implicit type coercion behavior, where the types of some values are changed automatically when compatible depending on the context. E.g. automatic dereferencing or downgrading a `&mut` into a `&`.

See [Type coercions | Reference](https://doc.rust-lang.org/reference/type-coercions.html).

## `tests/ui/coherence/`: Trait Implementation Coherence

Tests for trait coherence rules, which govern where trait implementations can be defined. E.g. orphan rule, and overlap checks.

See [Coherence | rustc-dev-guide](https://rustc-dev-guide.rust-lang.org/coherence.html#coherence).

## `tests/ui/coinduction/`: Coinductive Trait Resolution

Tests for coinduction in trait solving which may involve infinite proof trees.

See:

- [Coinduction | rustc-dev-guide](https://rustc-dev-guide.rust-lang.org/solve/coinduction.html).
- [Inductive cycles | Chalk](https://rust-lang.github.io/chalk/book/recursive/inductive_cycles.html#inductive-cycles)/

This directory only contains one highly specific test. Other coinduction tests can be found down the deeply located `tests/ui/traits/next-solver/cycles/coinduction/` subdirectory.

## `tests/ui/command/`: `std::process::Command`

This directory is actually for the standard library [`std::process::Command`](https://doc.rust-lang.org/std/process/struct.Command.html) type, where some tests are too difficult or inconvenient to write as unit tests or integration tests within the standard library itself.

**FIXME**: the test `command-line-diagnostics` seems to have been misplaced in this category.

## `tests/ui/compare-method/`: Trait implementation and definition comparisons

Some traits' implementation must be compared with their definition, checking for problems such as the implementation having stricter requirements (such as needing to implement `Copy`).

This subdirectory is *not* intended comparison traits (`PartialEq`, `Eq`, `PartialOrd`, `Ord`).

## `tests/ui/compiletest-self-test/`: compiletest "meta" tests

Meta test suite of the test harness `compiletest` itself.

## `tests/ui/conditional-compilation/`: Conditional Compilation

Tests for `#[cfg]` attribute or `--cfg` flags, used to compile certain files or code blocks only if certain conditions are met (such as developing on a specific architecture).

**FIXME**: There is significant overlap with `tests/ui/cfg`, which even contains a `tests/ui/cfg/conditional-compile.rs` test. Also investigate `tests/ui/check-cfg`.

## `tests/ui/confuse-field-and-method/`: Field/Method Ambiguity

If a developer tries to create a `struct` where one of the fields is a closure function, it becomes unclear whether `struct.field()` is accessing the field itself or trying to call the closure function within as a method.

**FIXME**: does this really need to be its own immediate subdirectory?

## `tests/ui/const-generics/`: Constant Generics

Tests for const generics, allowing types to be parameterized by constant values. It is generally observed in the form `<const N: Type>` after the `fn` or `struct` keywords. Includes tests for const expressions in generic contexts and associated type bounds.

See:

- [Tracking Issue for complex generic constants: `feature(generic_const_exprs)` #76560](https://github.com/rust-lang/rust/issues/76560)
- [Const generics | Reference](https://doc.rust-lang.org/reference/items/generics.html#const-generics)

## `tests/ui/const_prop/`: Constant Propagation

Tests exercising `ConstProp` mir-opt pass (mostly regression tests). See <https://blog.rust-lang.org/inside-rust/2019/12/02/const-prop-on-by-default/>.

## `tests/ui/const-ptr/`: Constant Pointers

Tests exercise const raw pointers. E.g. pointer arithmetic, casting and dereferencing, always with a `const`.

See:

- [`std::primitive::pointer`](https://doc.rust-lang.org/std/primitive.pointer.html)
- [`std::ptr`](https://doc.rust-lang.org/std/ptr/index.html)
- [Pointer types | Reference](https://doc.rust-lang.org/reference/types/pointer.html)

## `tests/ui/consts/`: General Constant Evaluation

Anything to do with constants, which does not fit in the previous two `const` categories, goes here. This does not always imply use of the `const` keyword - other values considered constant, such as defining an enum variant as `enum Foo { Variant = 5 }` also counts.

## `tests/ui/contracts/`: Contracts feature

Tests exercising `#![feature(contracts)]`.

See [Tracking Issue for Contracts #128044](https://github.com/rust-lang/rust/issues/128044).

## `tests/ui/coroutine/`: Coroutines feature and `gen` blocks

Tests for `#![feature(coroutines)]` and `gen` blocks, it belongs here.

See:

- [Coroutines | The Unstable book](https://doc.rust-lang.org/beta/unstable-book/language-features/coroutines.html)
- [RFC 3513 Gen blocks](https://rust-lang.github.io/rfcs/3513-gen-blocks.html)

## `tests/ui/coverage-attr/`: `#[coverage]` attribute

Tests for `#![feature(coverage_attribute)]`. See [Tracking issue for function attribute `#[coverage]`](https://github.com/rust-lang/rust/issues/84605).

## `tests/ui/crate-loading/`: Crate Loading

Tests for crate resolution and loading behavior, including `extern crate` declarations, `--extern` flags, or the `use` keyword.

## `tests/ui/cross/`: Various tests related to the concept of "cross"

**FIXME**: The unifying topic of these tests appears to be that their filenames begin with the word "cross". The similarities end there - one test is about "cross-borrowing" a `Box<T>` into `&T`, while another is about a global trait used "across" files. Some of these terminology are really outdated and does not match the current terminology. Additionally, "cross" is also way too generic, it's easy to confuse with cross-compile.

## `tests/ui/cross-crate/`: Cross-Crate Interaction

Tests for behavior spanning multiple crates, including visibility rules, trait implementations, and type resolution across crate boundaries.

## `tests/ui/custom_test_frameworks/`

Tests for `#[bench]`, `#[test_case]` attributes and the `custom_test_frameworks` lang item.

See [Tracking issue for eRFC 2318, Custom test frameworks #50297](https://github.com/rust-lang/rust/issues/50297).

## `tests/ui/c-variadic/`: C Variadic Function

Tests for FFI with C varargs (`va_list`).

## `tests/ui/cycle-trait/`: Trait Cycle Detection

Tests for detection and handling of cyclic trait dependencies.

## `tests/ui/dataflow_const_prop/`

Contains a single regression test for const prop in `SwitchInt` pass crashing when `ptr2int` transmute is involved.

**FIXME**: A category with a single test. Maybe it would fit inside the category `const-prop` or some kind of `mir-opt` directory.

## `tests/ui/debuginfo/`

Tests for generation of debug information (DWARF, etc.) including variable locations, type information, and source line mapping. Also exercises `-C split-debuginfo` and `-C debuginfo`.

## `tests/ui/definition-reachable/`: Definition Reachability

Tests to check whether definitions are reachable.

## `tests/ui/delegation/`: `#![feature(fn_delegation)]`

Tests for `#![feature(fn_delegation)]`. See [Implement function delegation in rustc #3530](https://github.com/rust-lang/rfcs/pull/3530) for the proposed prototype experimentation.

## `tests/ui/dep-graph/`: `-Z query-dep-graph`

These tests use the unstable command line option `query-dep-graph` to examine the dependency graph of a Rust program, which is useful for debugging.

## `tests/ui/deprecation/`: Deprecation Attribute

Tests for `#[deprecated]` attribute and `deprecated_in_future` internal lint.

## `tests/ui/deref-patterns/`: `#![feature(deref_patterns)]` and `#![feature(string_deref_patterns)]`

Tests for `#![feature(deref_patterns)]` and `#![feature(string_deref_patterns)]`. See [Deref patterns | The Unstable book](https://doc.rust-lang.org/nightly/unstable-book/language-features/deref-patterns.html).

**FIXME**: May have some overlap with `tests/ui/pattern/deref-patterns`.

## `tests/ui/derived-errors/`: Derived Error Messages

Tests for quality of diagnostics involving suppression of cascading errors in some cases to avoid overwhelming the user.

## `tests/ui/derives/`: Derive Macro

Tests for built-in derive macros (`Debug`, `Clone`, etc.) when used in conjunction with built-in `#[derive(..)]` attributes.

## `tests/ui/deriving/`: Derive Macro

**FIXME**: Coalesce with `tests/ui/derives`.

## `tests/ui/dest-prop/` Destination Propagation

**FIXME**: Contains a single test for the `DestProp` mir-opt, should probably be rehomed.

## `tests/ui/destructuring-assignment/`

Exercises destructuring assignments. See [RFC 2909 Destructuring assignment](https://github.com/rust-lang/rfcs/blob/master/text/2909-destructuring-assignment.md).

## `tests/ui/diagnostic-flags/`

These tests revolve around command-line flags which change the way error/warning diagnostics are emitted. For example, `--error-format=human --color=always`.

**FIXME**: Check redundancy with `annotate-snippet`, which is another emitter.

## `tests/ui/diagnostic_namespace/`

Exercises `#[diagnostic::*]` namespaced attributes. See [RFC 3368 Diagnostic attribute namepsace](https://github.com/rust-lang/rfcs/blob/master/text/3368-diagnostic-attribute-namespace.md).

## `tests/ui/diagnostic-width/`: `--diagnostic-width`

Everything to do with `--diagnostic-width`.

## `tests/ui/did_you_mean/`

Tests for miscellaneous suggestions.

## `tests/ui/directory_ownership/`: Declaring `mod` inside a block

Exercises diagnostics for when a code block attempts to gain ownership of a non-inline module with a `mod` keyword placed inside of it.

## `tests/ui/disallowed-deconstructing/`: Incorrect struct deconstruction

Exercises diagnostics for disallowed struct destructuring.

## `tests/ui/dollar-crate/`: `$crate` used with the `use` keyword

There are a few rules - which are checked in this directory - to follow when using `$crate` - it must be used in the start of a `use` line and is a reserved identifier.

**FIXME**: There are a few other tests in other directories with a filename starting with `dollar-crate`. They should perhaps be redirected here.

## `tests/ui/drop/`: `Drop` and drop order

Not necessarily about `Drop` and its implementation, but also about the drop order of fields inside a struct.

## `tests/ui/drop-bounds/`

Tests for linting on bounding a generic type on `Drop`.

## `tests/ui/dropck/`: Drop Checking

Mostly about checking the validity of `Drop` implementations.

See:

- [Dropck | The Nomicon](https://doc.rust-lang.org/nomicon/dropck.html)
- [Drop check | rustc-dev-guide](https://rustc-dev-guide.rust-lang.org/borrow_check/drop_check.html)

## `tests/ui/dst/`: Dynamically Sized Types

Tests for dynamically-sized types (DSTs). See [Dynamically Sized Types | Reference](https://doc.rust-lang.org/reference/dynamically-sized-types.html).

## `tests/ui/duplicate/`: Duplicate Symbols

Tests about duplicated symbol names and associated errors, such as using the `#[export_name]` attribute to rename a function with the same name as another function.

## `tests/ui/dynamically-sized-types/`: Dynamically Sized Types

**FIXME**: should be coalesced with `tests/ui/dst`.

## `tests/ui/dyn-compatibility/`: Dyn-compatibility

Tests for dyn-compatibility of traits.

See:

- [Trait object | Reference](https://doc.rust-lang.org/reference/types/trait-object.html)
- [Dyn compatibility | Reference](https://doc.rust-lang.org/reference/items/traits.html#dyn-compatibility)

Previously known as "object safety".

## `tests/ui/dyn-drop/`: `dyn Drop`

**FIXME**: Contains a single test, used only to check the `dyn_drop` lint (which is normally `warn` level).

## `tests/ui/dyn-keyword/`: `dyn` and Dynamic Dispatch

The `dyn` keyword is used to highlight that calls to methods on the associated Trait are dynamically dispatched. To use the trait this way, it must be dyn-compatible - tests about dyn-compatibility belong in `tests/ui/dyn-compatibility/`, while more general tests on dynamic dispatch belong here.

See [`dyn` keyword](https://doc.rust-lang.org/std/keyword.dyn.html).

## `tests/ui/dyn-star/`: `dyn*`, Sized `dyn`, `#![feature(dyn_star)]`

See [Tracking issue for dyn-star #102425](https://github.com/rust-lang/rust/issues/102425).

## `tests/ui/editions/`: Rust edition-specific peculiarities

These tests run in specific Rust editions, such as Rust 2015 or Rust 2018, and check errors and functionality related to specific now-deprecated idioms and features.

**FIXME**: Maybe regroup `rust-2018`, `rust-2021` and `rust-2024` under this umbrella?

## `tests/ui/empty/`: Various tests related to the concept of "empty"

**FIXME**: These tests need better homes, this is not very informative.

## `tests/ui/entry-point/`: `main` function

Tests exercising the `main` entry-point.

## `tests/ui/enum/`

General-purpose tests on `enum`s. See [Enumerations | Reference](https://doc.rust-lang.org/reference/items/enumerations.html).

## `tests/ui/enum-discriminant/`

`enum` variants can be differentiated independently of their potential field contents with `discriminant`, which returns the type `Discriminant<T>`. See [`std::mem::discriminant`](https://doc.rust-lang.org/std/mem/fn.discriminant.html).

## `tests/ui/env-macro/`: `env!`

Exercises `env!` and `option_env!` macros.

## `tests/ui/ergonomic-clones/`

See [RFC 3680 Ergonomic clones](https://github.com/rust-lang/rfcs/pull/3680).

## `tests/ui/error-codes/`: Error codes

Tests for errors with dedicated error codes.

## `tests/ui/error-emitter/`

Quite similar to `ui/diagnostic-flags` in some of its tests, this category checks some behaviours of Rust's error emitter into the user's terminal window, such as truncating error in the case of an excessive amount of them.

## `tests/ui/errors/`

These tests are about very different topics, only unified by the fact that they result in errors.

**FIXME**: "Errors" is way too generic, the tests probably need to be rehomed into more descriptive subdirectories.

## `tests/ui/explain/`: `rustc --explain EXXXX`

Accompanies `tests/ui/error-codes/`, exercises the `--explain` cli flag.

## `tests/ui/explicit/`: Errors involving the concept of "explicit"

This category contains three tests: two which are about the specific error `explicit use of destructor method`, and one which is about explicit annotation of lifetimes: https://doc.rust-lang.org/stable/rust-by-example/scope/lifetime/explicit.html.

**FIXME**: Rehome the two tests about the destructor method with `drop`-related categories, and rehome the last test with a category related to lifetimes.

## `tests/ui/explicit-tail-calls/`

Exercises `#![feature(explicit_tail_calls)]` and the `become` keyword. See [Explicit Tail Calls #3407](https://github.com/rust-lang/rfcs/pull/3407).

## `tests/ui/expr/`: Expressions

A broad directory for tests on expressions.

## `tests/ui/extern/`

Tests on the `extern` keyword and `extern` blocks and functions.

## `tests/ui/extern-flag/`: `--extern` command line flag

Tests for the `--extern` CLI flag.

## `tests/ui/feature-gates/`

Tests on feature-gating, and the `#![feature(..)]` mechanism itself.

## `tests/ui/ffi-attrs/`: `#![feature(ffi_const, ffi_pure)]`

The `#[ffi_const]` and `#[ffi_pure]` attributes applies clang's `const` and `pure` attributes to foreign functions declarations, respectively. These attributes are the core element of the tests in this category.

See:

- [`ffi_const` | The Unstable book](https://doc.rust-lang.org/unstable-book/language-features/ffi-const.html)
- [`ffi_pure` | The Unstable book](https://doc.rust-lang.org/beta/unstable-book/language-features/ffi-pure.html)

## `tests/ui/fmt/`

Exercises the `format!` macro.

## `tests/ui/fn/`

A broad category of tests on functions.

## `tests/ui/fn-main/`

**FIXME**: Serves a duplicate purpose with `ui/entry-point`, should be combined.

## `tests/ui/for/`: `for` keyword

Tests on the `for` keyword and some of its associated errors, such as attempting to write the faulty pattern `for _ in 0..1 {} else {}`.

**FIXME**: Should be merged with `ui/for-loop-while`.

## `tests/ui/force-inlining/`: `#[rustc_force_inline]`

Tests for `#[rustc_force_inline]`, which will force a function to always be labelled as inline by the compiler (it will be inserted at the point of its call instead of being used as a normal function call.) If the compiler is unable to inline the function, an error will be reported. See <https://github.com/rust-lang/rust/pull/134082>.

## `tests/ui/foreign/`: Foreign Function Interface (FFI)

Tests for `extern "C"` and `extern "Rust`.

**FIXME**: Check for potential overlap/merge with `ui/c-variadic` and/or `ui/extern`.

## `tests/ui/for-loop-while/`

Anything to do with loops and `for`, `loop` and `while` keywords to express them.

**FIXME**: After `ui/for` is merged into this, also carry over its SUMMARY text.

## `tests/ui/frontmatter/`

Tests for `#![feature(frontmatter)]`. See [Tracking Issue for `frontmatter` #136889](https://github.com/rust-lang/rust/issues/136889).

## `tests/ui/fully-qualified-type/`

Tests for diagnostics when there may be identically named types that need further qualifications to disambiguate.

## `tests/ui/functional-struct-update/`

Functional Struct Update is the name for the idiom by which one can write `..<expr>` at the end of a struct literal expression to fill in all remaining fields of the struct literal by using `<expr>` as the source for them.

See [RFC 0736 Privacy-respecting Functional Struct Update](https://github.com/rust-lang/rfcs/blob/master/text/0736-privacy-respecting-fru.md).

## `tests/ui/function-pointer/`

Tests on function pointers, such as testing their compatibility with higher-ranked trait bounds.

See:

- [Function pointer types | Reference](https://doc.rust-lang.org/reference/types/function-pointer.html)
- [Higher-ranked trait bounds | Nomicon](https://doc.rust-lang.org/nomicon/hrtb.html)

## `tests/ui/functions-closures/`

Tests on closures. See [Closure expressions | Reference](https://doc.rust-lang.org/reference/expressions/closure-expr.html).

## `tests/ui/generic-associated-types/`

Tests on Generic Associated Types (GATs).

## `tests/ui/generic-const-items/`

Tests for `#![feature(generic_const_items)]`. See [Tracking issue for generic const items #113521](https://github.com/rust-lang/rust/issues/113521).

## `tests/ui/generics/`

A broad category of tests on generics, usually used when no more specific subdirectories are fitting.

## `tests/ui/half-open-range-patterns/`: `x..` or `..x` range patterns

Tests on range patterns where one of the bounds is not a direct value.

**FIXME**: Overlaps with `ui/range`. `impossible_range.rs` is particularly suspected to be a duplicate test.

## `tests/ui/hashmap/`

Tests for the standard library collection [`std::collections::HashMap`](https://doc.rust-lang.org/std/collections/struct.HashMap.html).

## `tests/ui/hello_world/`

Tests that the basic hello-world program is not somehow broken.

## `tests/ui/higher-ranked/`

Tests for higher-ranked trait bounds.

See:

- [Higher-ranked trait bounds | rustc-dev-guide](https://rustc-dev-guide.rust-lang.org/traits/hrtb.html)
- [Higher-ranked trait bounds | Nomicon](https://doc.rust-lang.org/nomicon/hrtb.html)

## `tests/ui/hygiene/`

This seems to have been originally intended for "hygienic macros" - macros which work in all contexts, independent of what surrounds them. However, this category has grown into a mish-mash of many tests that may belong in the other directories.

**FIXME**: Reorganize this directory properly.

## `tests/ui/illegal-sized-bound/`

This test category revolves around trait objects with `Sized` having illegal operations performed on them.

**FIXME**: There seems to be unrelated testing in this directory, such as `tests/ui/illegal-sized-bound/mutability-mismatch-arg.rs`. Investigate.

## `tests/ui/impl-header-lifetime-elision/`

Tests on lifetime elision in impl function signatures. See [Lifetime elision | Nomicon](https://doc.rust-lang.org/nomicon/lifetime-elision.html).

## `tests/ui/implied-bounds/`

See [Implied bounds | Reference](https://doc.rust-lang.org/reference/trait-bounds.html#implied-bounds).

## `tests/ui/impl-trait/`

Tests for trait impls.

## `tests/ui/imports/`

Tests for module system and imports.

## `tests/ui/include-macros/`

Exercise `include!`, `include_str!`, and `include_bytes!` macros.

## `tests/ui/incoherent-inherent-impls/`

Exercise forbidding inherent impls on a type defined in a different crate.

## `tests/ui/indexing/`

Tests on collection types (arrays, slices, vectors) and various errors encountered when indexing their contents, such as accessing out-of-bounds values.

**FIXME**: (low-priority) could maybe be a subdirectory of `ui/array-slice-vec`

## `tests/ui/inference/`

Tests on type inference.

## `tests/ui/infinite/`

Tests for diagnostics on infinitely recursive types without indirection.

## `tests/ui/inherent-impls-overlap-check/`

Checks that repeating the same function names across separate `impl` blocks triggers an informative error, but not if the `impl` are for different types, such as `Bar<u8>` and `Bar<u16>`.

NOTE: This should maybe be a subdirectory within another related to duplicate definitions, such as `tests/ui/duplicate/`.

## `tests/ui/inline-const/`

These tests revolve around the inline `const` block that forces the compiler to const-eval its content.

## `tests/ui/instrument-coverage/`: `-Cinstrument-coverage` command line flag

See [Instrument coverage | The rustc book](https://doc.rust-lang.org/rustc/instrument-coverage.html).

## `tests/ui/instrument-xray/`: `-Z instrument-xray`

See [Tracking issue for `-Z instrument-xray` #102921](https://github.com/rust-lang/rust/issues/102921).

## `tests/ui/interior-mutability/`

**FIXME**: contains a single test, probably better rehomed.

## `tests/ui/internal/`

Tests for `internal_unstable` and the attribute header `#![feature(allow_internal_unstable)]`, which lets compiler developers mark features as internal to the compiler, and unstable for standard library use.

## `tests/ui/internal-lints/`

Tests for rustc-internal lints.

## `tests/ui/intrinsics/`

Tests for the `{std,core}::intrinsics`, internal implementation detail.

## `tests/ui/invalid/`

Various tests related to rejecting invalid inputs.

**FIXME**: This is rather uninformative, possibly rehome into more meaningful directories.

## `tests/ui/invalid-compile-flags/`

Tests for checking that invalid usage of compiler flags are rejected.

## `tests/ui/invalid-module-declaration/`

**FIXME**: Consider merging into module/resolve directories.

## `tests/ui/invalid-self-argument/`: `self` as a function argument incorrectly

Tests with erroneous ways of using `self`, such as having it not be the first argument, or using it in a non-associated function (no `impl` or `trait`).

**FIXME**: Maybe merge with `ui/self`.

## `tests/ui/io-checks/`

Contains a single test. The test tries to output a file into an invalid directory with `-o`, then checks that the result is an error, not an internal compiler error.

**FIXME**: Rehome to invalid compiler flags maybe.

## `tests/ui/issues/`: Tests directly related to GitHub issues

**FIXME (#133895)**: Random collection of regression tests and tests for issues, tests in this directory should be audited and rehomed.

## `tests/ui/iterators/`

These tests revolve around anything to do with iterators, e.g. mismatched types.

**FIXME**: Check for potential overlap with `ui/for-loop-while`.

## `tests/ui/json/`

These tests revolve around the `--json` compiler flag. See [JSON Output](https://doc.rust-lang.org/rustc/json.html#json-output).

## `tests/ui/keyword/`

Tests exercising keywords, such as attempting to use them as identifiers when not contextual keywords.

## `tests/ui/kindck/`

**FIXME**: `kindck` is no longer a thing, these tests probably need to be audited and rehomed.

## `tests/ui/label/`

Exercises block and loop `'label`s.

## `tests/ui/lang-items/`

See [Lang items | The Unstable book](https://doc.rust-lang.org/unstable-book/language-features/lang-items.html).

## `tests/ui/late-bound-lifetimes/`

See [Early vs Late bound parameters | rustc-dev-guide](https://rustc-dev-guide.rust-lang.org/early_late_parameters.html#early-vs-late-bound-parameters).

## `tests/ui/layout/`

See [Type Layout | Reference](https://doc.rust-lang.org/reference/type-layout.html).

## `tests/ui/lazy-type-alias/`

Tests for `#![feature(lazy_type_alias)]`. See [Tracking issue for lazy type aliases #112792
](https://github.com/rust-lang/rust/issues/112792).

## `tests/ui/lazy-type-alias-impl-trait/`

This feature allows use of an `impl Trait` in multiple locations while actually using the same concrete type (`type Alias = impl Trait;`) everywhere, keeping the original `impl Trait` hidden.

**FIXME**: merge this with `tests/ui/type-alias-impl-trait/`?

## `tests/ui/let-else/`

Exercises let-else constructs.

## `tests/ui/lexer/`

Exercises of the lexer.

## `tests/ui/lifetimes/`

Broad directory on lifetimes, including proper specifiers, lifetimes not living long enough, or undeclared lifetime names.

## `tests/ui/limits/`

These tests exercises numerical limits, such as `[[u8; 1518599999]; 1518600000]`.

## `tests/ui/linkage-attr/`

Tests for the linkage attribute `#[linkage]` of `#![feature(linkage)]`.

**FIXME**: Some of these tests do not use the feature at all, which should be moved under `ui/linking` instead.

## `tests/ui/linking/`

Tests on code which fails during the linking stage, or which contain arguments and lines that have been known to cause unjustified errors in the past, such as specifying an unusual `#[export_name]`.

See [Linkage | Reference](https://doc.rust-lang.org/reference/linkage.html).

## `tests/ui/link-native-libs/`

Tests for `#[link(name = "", kind = "")]` and `-l` command line flag.

See [Tracking Issue for linking modifiers for native libraries #81490](https://github.com/rust-lang/rust/issues/81490).

## `tests/ui/lint/`

Tests for the lint infrastructure, lint levels, lint reasons, etc.

See:

- [Lints | The rustc book](https://doc.rust-lang.org/rustc/lints/index.html)
- [Lint reasons | Reference](https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-reasons)

## `tests/ui/liveness/`

Tests exercising analysis for unused variables, unreachable statements, functions which are supposed to return a value but do not, as well as values moved elsewhere before they could be used by a function.

**FIXME**: This seems unrelated to "liveness" as defined in the rustc compiler guide. Is this misleadingly named? https://rustc-dev-guide.rust-lang.org/borrow_check/region_inference/lifetime_parameters.html#liveness-and-universal-regions

## `tests/ui/loops/`

Tests on the `loop` construct.

**FIXME**: Consider merging with `ui/for-loop-while`.

## `tests/ui/lowering/`

Tests on [AST lowering](https://rustc-dev-guide.rust-lang.org/ast-lowering.html).

## `tests/ui/lto/`

Exercise *Link-Time Optimization* (LTO), involving the flags `-C lto` or `-Z thinlto`.

## `tests/ui/lub-glb/`: LUB/GLB algorithm update

Tests on changes to inference variable lattice LUB/GLB, see <https://github.com/rust-lang/rust/pull/45853>.

## `tests/ui/macro_backtrace/`: `-Zmacro-backtrace`

Contains a single test, checking the unstable command-line flag to enable detailed macro backtraces.

**FIXME**: This could be merged with `ui/macros`, which already contains other macro backtrace tests.

## `tests/ui/macros/`

Broad category of tests on macros.

## `tests/ui/malformed/`

Broad category of tests on malformed inputs.

**FIXME**: this is kinda vague, probably best to audit and rehome tests.

## `tests/ui/marker_trait_attr/`

See [Tracking issue for allowing overlapping implementations for marker trait #29864](https://github.com/rust-lang/rust/issues/29864).

## `tests/ui/match/`

Broad category of tests on `match` constructs.

## `tests/ui/meta/`: Tests for compiletest itself

These tests check the function of the UI test suite at large and Compiletest in itself.

**FIXME**: This should absolutely be merged with `tests/ui/compiletest-self-test/`.

## `tests/ui/methods/`

A broad category for anything related to methods and method resolution.

## `tests/ui/mir/`

Certain mir-opt regression tests.

## `tests/ui/mir-dataflow`

Tests for MIR dataflow analysis.

See [MIR Dataflow | rustc-dev-guide](https://rustc-dev-guide.rust-lang.org/mir/dataflow.html).

## `tests/ui/mismatched_types/`

Exercises on mismatched type diagnostics.

## `tests/ui/missing/`

Something is missing which could be added to fix (e.g. suggestions).

**FIXME**: this is way too vague, tests should be rehomed.

## `tests/ui/missing_non_modrs_mod/`

This directory is a small tree of `mod` dependencies, but the root, `foo.rs`, is looking for a file which does not exist. The test checks that the error is reported at the top-level module.

**FIXME**: Merge with `tests/ui/modules/`.

## `tests/ui/missing-trait-bounds/`

Tests for checking missing trait bounds, and their diagnostics.

**FIMXE**: Maybe a subdirectory of `ui/trait-bounds` would be more appropriate.

## `tests/ui/modules/`

Tests on the module system.

**FIXME**: `tests/ui/imports/` should probably be merged with this.

## `tests/ui/modules_and_files_visibility/`

**FIXME**: Merge with `tests/ui/modules/`.

## `tests/ui/moves`

Tests on moves (destructive moves).

## `tests/ui/mut/`

Broad category of tests on mutability, such as the `mut` keyword, borrowing a value as both immutable and mutable (and the associated error), or adding mutable references to `const` declarations.

## `tests/ui/namespace/`

Contains a single test. It imports a massive amount of very similar types from a crate, then attempts various permutations of their namespace paths, checking for errors or the lackthereof.

**FIXME**: Move under either `tests/ui/modules/` or `tests/ui/resolve/`.

## `tests/ui/never_type/`

See [Tracking issue for promoting `!` to a type (RFC 1216) #35121](https://github.com/rust-lang/rust/issues/35121).

## `tests/ui/new-range/`

See [RFC 3550 New Range](https://github.com/rust-lang/rfcs/blob/master/text/3550-new-range.md).

## `tests/ui/nll/`: Non-lexical lifetimes

Tests for Non-lexical lifetimes. See [RFC 2094 NLL](https://rust-lang.github.io/rfcs/2094-nll.html).

## `tests/ui/non_modrs_mods/`

Despite the size of the directory, this is a single test, spawning a sprawling `mod` dependency tree and checking its successful build.

**FIXME**: Consider merge with `tests/ui/modules/`, keeping the directory structure.

## `tests/ui/non_modrs_mods_and_inline_mods/`

A very similar principle as `non_modrs_mods`, but with an added inline `mod` statement inside another `mod`'s code block.

**FXIME**: Consider merge with `tests/ui/modules/`, keeping the directory structure.

## `tests/ui/no_std/`

Tests for where the standard library is disabled through `#![no_std]`.

## `tests/ui/not-panic/`

Tests checking various types, such as `&RefCell<i32>`, and whether they are not `UnwindSafe` as expected.

## `tests/ui/numbers-arithmetic/`

Tests that exercises edge cases, such as specific floats, large or very small numbers, or bit conversion, and check that the arithmetic results are as expected.

## `tests/ui/numeric/`

Tests that checks numeric types and their interactions, such as casting among them with `as` or providing the wrong numeric suffix.

## `tests/ui/object-lifetime/`

Tests on lifetimes on objects, such as a lifetime bound not being able to be deduced from context, or checking that lifetimes are inherited properly.

**FIXME**: Just a more specific subset of `ui/lifetimes`.

## `tests/ui/obsolete-in-place/`

Contains a single test. Check that we reject the ancient Rust syntax `x <- y` and `in(BINDING) {}` construct.

**FIXME**: Definitely should be rehomed, maybe to `tests/ui/deprecation/`.

## `tests/ui/offset-of/`

Exercises the [`std::mem::offset_of` macro](https://doc.rust-lang.org/beta/std/mem/macro.offset_of.html).

## `tests/ui/on-unimplemented/`

Exercises the `#[rustc_on_unimplemented]`.

## `tests/ui/operator-recovery/`

**FIXME**: Probably move under `tests/ui/binop/` or `tests/ui/parser/`.

## `tests/ui/or-patterns/`

Exercises `||` and `|` in patterns.

## `tests/ui/overloaded/`

Exercises operator overloading via [`std::ops`](https://doc.rust-lang.org/std/ops/index.html).

## `tests/ui/packed/`

See [`repr(packed)` | Nomicon](https://doc.rust-lang.org/nomicon/other-reprs.html#reprpacked-reprpackedn).

## `tests/ui/panic-handler/`

See [panic handler | Nomicon](https://doc.rust-lang.org/nomicon/panic-handler.html).

## `tests/ui/panic-runtime/`

Exercises `#![panic_runtime]`, `-C panic`, panic runtimes and panic unwind strategy.

See [RFC 1513 Less unwinding](https://github.com/rust-lang/rfcs/blob/master/text/1513-less-unwinding.md).

## `tests/ui/panics/`

Broad category of tests about panics in general, often but not necessarily using the `panic!` macro.

## `tests/ui/parallel-rustc/`

Efforts towards a [Parallel Rustc Front-end](https://github.com/rust-lang/rust/issues/113349). Includes `-Zthreads=`.

## `tests/ui/parser/`

Various parser tests

**FIXME**: Maybe move `tests/ui/keywords/` under this?

## `tests/ui/patchable-function-entry/`

See [Patchable function entry | The Unstable book](https://doc.rust-lang.org/unstable-book/compiler-flags/patchable-function-entry.html).

## `tests/ui/pattern/`

Broad category of tests surrounding patterns. See [Patterns | Reference](https://doc.rust-lang.org/reference/patterns.html).

**FIXME**: Some overlap with `tests/ui/match/`.

## `tests/ui/pin-macro/`

See [`std::pin`](https://doc.rust-lang.org/std/pin/).

## `tests/ui/precondition-checks/`

Exercises on some unsafe precondition checks.

## `tests/ui/print-request/`

Tests on `--print` compiler flag. See [print options | The rustc book](https://doc.rust-lang.org/rustc/command-line-arguments/print-options.html).

## `tests/ui/print_type_sizes/`

Exercises the `-Z print-type-sizes` flag.

## `tests/ui/privacy/`

Exercises on name privacy. E.g. the meaning of `pub`, `pub(crate)`, etc.

## `tests/ui/process/`

Some standard library process tests which are hard to write within standard library crate tests.

## `tests/ui/process-termination/`

Some standard library process termination tests which are hard to write within standard library crate tests.

## `tests/ui/proc-macro/`

Broad category of tests on proc-macros. See [Procedural Macros | Reference](https://doc.rust-lang.org/reference/procedural-macros.html).

## `tests/ui/ptr_ops/`: Using operations on a pointer

Contains only 2 tests, related to a single issue, which was about an error caused by using addition on a pointer to `i8`.

**FIXME**: Probably rehome under some typecheck / binop directory.

## `tests/ui/pub/`: `pub` keyword

A large category about function and type public/private visibility, and its impact when using features across crates. Checks both visibility-related error messages and previously buggy cases.

**FIXME**: merge with `tests/ui/privacy/`.

## `tests/ui/qualified/`

Contains few tests on qualified paths where a type parameter is provided at the end: `type A = <S as Tr>::A::f<u8>;`. The tests check if this fails during type checking, not parsing.

**FIXME**: Should be rehomed to `ui/typeck`.

## `tests/ui/query-system/`

Tests on Rust methods and functions which use the query system, such as `std::mem::size_of`. These compute information about the current runtime and return it. See [Query system | rustc-dev-guide](https://rustc-dev-guide.rust-lang.org/query.html).

## `tests/ui/range/`

Broad category of tests ranges, both in their `..` or `..=` form, as well as the standard library `Range`, `RangeTo`, `RangeFrom` or `RangeBounds` types.

**FIXME**: May have some duplicate tests with `ui/new-range`.

## `tests/ui/raw-ref-op/`: Using operators on `&raw` values

Exercises `&raw mut <place>` and `&raw const <place>`. See [RFC 2582 Raw reference MIR operator](https://github.com/rust-lang/rfcs/blob/master/text/2582-raw-reference-mir-operator.md).

## `tests/ui/reachable`

Reachability tests, primarily unreachable code and coercions into the never type `!` from diverging expressions.

**FIXME**: Check for overlap with `ui/liveness`.

## `tests/ui/recursion/`

Broad category of tests exercising recursions (compile test and run time), in functions, macros, `type` definitions, and more.

Also exercises the `#![recursion_limit = ""]` attribute.

## `tests/ui/recursion_limit/`: `#![recursion_limit = ""]`

Sets a recursion limit on recursive code.

**FIXME**: Should be merged with `tests/ui/recursion/`.

## `tests/ui/regions/`

**FIXME**: Maybe merge with `ui/lifetimes`.

## `tests/ui/repeat-expr/`

Exercises `[Type; n]` syntax for creating arrays with repeated types across a set size.

**FIXME**: Maybe make this a subdirectory of `ui/array-slice-vec`.

## `tests/ui/repr/`: `#[repr(_)]`

Tests on the `#[repr(..)]` attribute. See [Representations | Reference](https://doc.rust-lang.org/reference/type-layout.html#representations).

## `tests/ui/reserved/`

Reserved keywords and attribute names.

See e.g. [Reserved keywords | Reference](https://doc.rust-lang.org/reference/keywords.html).

**FIXME**: maybe merge under `tests/ui/keyword/`.

## `tests/ui/resolve/`: Name resolution

See [Name resolution | rustc-dev-guide](https://rustc-dev-guide.rust-lang.org/name-resolution.html).

## `tests/ui/return/`

Exercises the `return` keyword, return expressions and statements.

## `tests/ui/rfcs/`

Tests that accompanies an implementation for an RFC.

## `tests/ui/rmeta/`

Exercises `.rmeta` crate metadata and the `--emit=metadata` cli flag.

## `tests/ui/runtime/`

Tests for runtime environment on which Rust programs are executed. E.g. Unix `SIGPIPE`.

## `tests/ui/rust-{2018,2021,2024}/`

Tests that exercise behaviors and features that are specific to editions.

## `tests/ui/rustc-env`

Tests on environmental variables that affect `rustc`.

## `tests/ui/rustdoc`

Hybrid tests that exercises `rustdoc`, and also some joint `rustdoc`/`rustc` interactions.

## `tests/ui/sanitizer/`

Exercises sanitizer support. See [Sanitizer | The rustc book](https://doc.rust-lang.org/unstable-book/compiler-flags/sanitizer.html).

## `tests/ui/self/`: `self` keyword

Tests with erroneous ways of using `self`, such as using `this.x` syntax as seen in other languages, having it not be the first argument, or using it in a non-associated function (no `impl` or `trait`). It also contains correct uses of `self` which have previously been observed to cause ICEs.

## `tests/ui/sepcomp/`: Separate Compilation

In this directory, multiple crates are compiled, but some of them have `inline` functions, meaning they must be inlined into a different crate despite having been compiled separately.

**FIXME**: this directory might need some better docs, also this directory might want a better name.

## `tests/ui/shadowed/`

Tests on name shadowing.

## `tests/ui/shell-argfiles/`: `-Z shell-argfiles` command line flag

The `-Zshell-argfiles` compiler flag allows argfiles to be parsed using POSIX "shell-style" quoting. When enabled, the compiler will use shlex to parse the arguments from argfiles specified with `@shell:<path>`.

Because this feature controls the parsing of input arguments, the `-Zshell-argfiles` flag must be present before the argument specifying the shell-style argument file.

**FIXME**: maybe group this with `tests/ui/argfile/`

## `tests/ui/simd/`

Some tests exercising SIMD support.

## `tests/ui/single-use-lifetime/`

This is a test directory for the specific error case where a lifetime never gets used beyond a single annotation on, for example, a `struct`.

## `tests/ui/sized/`: `Sized` trait, sized types

While many tests here involve the `Sized` trait directly, some instead test, for example the slight variations between returning a zero-sized `Vec` and a `Vec` with one item, where one has no known type and the other does.

## `tests/ui/span/`

An assorted collection of tests that involves specific diagnostic spans.

**FIXME**: This is a big directory with numerous only-tangentially related tests. Maybe some moving is in order.

## `tests/ui/specialization`

See [Tracking issue for specialization (RFC 1210) #31844](https://github.com/rust-lang/rust/issues/31844).

## `tests/ui/stability-attribute/`

Stability attributes used internally by the standard library: `#[stable()]` and `#[unstable()]`.

## `tests/ui/stable-mir-print/`

Some tests for pretty printing of StableMIR.

## `tests/ui/stack-protector/`: `-Z stack-protector` command line flag

See [Tracking Issue for stabilizing stack smashing protection (i.e., `-Z stack-protector`) #114903](https://github.com/rust-lang/rust/issues/114903).

## `tests/ui/static/`

Tests on static items.

## `tests/ui/statics/`

**FIXME**: should probably be merged with `tests/ui/static/`.

## `tests/ui/stats/`

Tests for compiler-internal stats; `-Z meta-stats` and `-Z input-stats` flags.

## `tests/ui/std/`: Tests which use features from the standard library

A catch-all category about anything that can come from `std`.

**FIXME**: this directory is probably too vague, tests might need to be audited and rehomed.

## `tests/ui/stdlib-unit-tests/`

Some standard library tests which are too inconvenient or annoying to implement as std crate tests.

## `tests/ui/str/`

Exercise `str` keyword and string slices.

## `tests/ui/structs/`

Assorted tests surrounding the `struct` keyword, struct type definitions and usages.

## `tests/ui/structs-enums/`

Tests on both structs and enums.

**FIXME**: maybe coalesce {`tests/ui/structs/`, `tests/ui/structs-enums/`, `tests/ui/enums/`} into one `tests/ui/adts` directory...

## `tests/ui/suggestions/`

Generic collection of tests for suggestions, when no more specific directories are applicable.

**FIXME**: Some overlap with `tests/ui/did_you_mean/`, that directory should probably be moved under here.

## `tests/ui/svh/`: Strict Version Hash

Tests on the *Strict Version Hash* (SVH, also known as the "crate hash").

See [Strict Version Hash](https://rustc-dev-guide.rust-lang.org/backend/libs-and-metadata.html#strict-version-hash).

## `tests/ui/symbol-mangling-version/`: `-Csymbol-mangling-version` command line flag

**FIXME**: Should be merged with `ui/symbol-names`.

## `tests/ui/symbol-names/`: Symbol mangling and related attributes

These tests revolve around `#[no_mangle]` attribute, as well as consistently mangled symbol names (checked with the `rustc_symbol_name` attribute), which is important to build reproducible binaries.

## `tests/ui/sync/`: `Sync` trait

Exercises `Sync` trait and auto-derive thereof.

## `tests/ui/target-cpu/`: `-C target-cpu` command line flag

This command line option instructs rustc to generate code specifically for a particular processor.

**FIXME**: Contains a single test, maybe put it in a directory about misc codegen options?

## `tests/ui/target-feature/`: `#[target_feature(enable = "relaxed-simd")]`

Exercises the `#[target_feature(..)]` attribute. See [Target feature attribute | Reference](https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute).

## `tests/ui/target_modifiers/`

Tests for [RFC 3716: Target Modifiers](https://github.com/rust-lang/rfcs/pull/3716).

See [Tracking Issue for target modifiers #136966](https://github.com/rust-lang/rust/issues/136966).

## `tests/ui/test-attrs/`

Exercises the [`#[test]` attribute](https://doc.rust-lang.org/reference/attributes/testing.html#testing-attributes).

## `tests/ui/thir-print/`

Pretty print of THIR trees via `-Zunpretty=thir-tree`.

## `tests/ui/thread-local/`

Exercises thread local values and `#[thread_local]` attribute.

See [Tracking issue for thread_local stabilization #29594](https://github.com/rust-lang/rust/issues/29594).

## `tests/ui/threads-sendsync/`

Broad category for parallelism and multi-threaded tests, including attempting to send types across threads which are not thread-safe.

## `tests/ui/tool-attributes/`: External tool attributes

Exercises [tool attributes](https://doc.rust-lang.org/reference/attributes.html#tool-attributes).

## `tests/ui/track-diagnostics/`

Exercises `#[track_caller]` and `-Z track-diagnostics`.

## `tests/ui/trait-bounds/`

Collection of tests for [trait bounds](https://doc.rust-lang.org/reference/trait-bounds.html).

## `tests/ui/traits/`

Broad collection of tests on traits in general.

**FIXME**: This could be better organized in subdirectories containing tests such as `ui/traits/trait-bounds`.

## `tests/ui/transmutability/`: `#![feature(transmutability)]`

See [Tracking Issue for Transmutability Trait: `#[transmutability]` #99571](https://github.com/rust-lang/rust/issues/99571).

See also [Project Safe Transmute](https://github.com/rust-lang/project-safe-transmute).

## `tests/ui/transmute/`

Tests surrounding [`std::mem::transmute`](https://doc.rust-lang.org/std/mem/fn.transmute.html).

## `tests/ui/treat-err-as-bug/`

Exercises compiler development support flag `-Z treat-err-as-bug`.

## `tests/ui/trivial-bounds/`

`#![feature(trivial_bounds)]`. See [RFC 2056 Allow trivial where clause constraints](https://github.com/rust-lang/rfcs/blob/master/text/2056-allow-trivial-where-clause-constraints.md).

## `tests/ui/try-block/`

`#![feature(try_blocks)]`. See [Tracking issue for `?` operator and `try` blocks (RFC 243, `question_mark` & `try_blocks` features)](https://github.com/rust-lang/rust/issues/31436).

## `tests/ui/try-trait/`

`#![feature(try_trait_v2)]`. See [RFC 3058 Try Trait v2](https://github.com/rust-lang/rfcs/blob/master/text/3058-try-trait-v2.md).

## `tests/ui/tuple/`

Tests surrounding the tuple type `()`.

## `tests/ui/type/`

Assorted collection of tests surrounding the concept of a "type".

**FIXME**: There is very little consistency across tests of this category, and should probably be sent to other subdirectories.

## `tests/ui/type-alias/`

Exercises [type aliases](https://doc.rust-lang.org/reference/items/type-aliases.html).

## `tests/ui/type-alias-enum-variants/`

Tests for `type` aliases in the context of `enum` variants, such as that applied type arguments of enums are respected independently of being the original type or the `type` alias.

## `tests/ui/type-alias-impl-trait/`

`#![feature(type_alias_impl_trait)]`. See [Type Alias Impl Trait | The Unstable book](https://doc.rust-lang.org/nightly/unstable-book/language-features/type-alias-impl-trait.html).

## `tests/ui/typeck/`

General collection of type checking related tests.

## `tests/ui/type-inference/`

General collection of type inference related tests.

## `tests/ui/typeof/`

`typeof` keyword, reserved but unimplemented.

## `tests/ui/ufcs/`

See [RFC 0132 Unified Function Call Syntax](https://github.com/rust-lang/rfcs/blob/master/text/0132-ufcs.md).

## `tests/ui/unboxed-closures/`

`#![feature(unboxed_closures)]`, `Fn`, `FnMut` and `FnOnce` traits

See [Tracking issue for Fn traits (`unboxed_closures` & `fn_traits` feature)](https://github.com/rust-lang/rust/issues/29625).

## `tests/ui/underscore-imports/`

See [Underscore imports | Reference](https://doc.rust-lang.org/reference/items/use-declarations.html#underscore-imports).

**FIXME**: should become a subdirectory of `tests/ui/imports/`.

## `tests/ui/underscore-lifetime/`: `'_` elided lifetime

Exercises [anonymous elided lifetimes](https://doc.rust-lang.org/reference/lifetime-elision.html).

## `tests/ui/uniform-paths/`

In uniform paths, if a submodule and an external dependencies have the same name, to depend on the external dependency, one needs to disambiguate it from the submodule using `use ::foo`. Tests revolve around this, for example, check that `self::foo` and `::foo` are not considered ambiguously identical by the compiler.

Remark: As they are an important Rust 2018 feature, they also get a big subdirectory in `ui/rust-2018/uniform-paths`

## `tests/ui/uninhabited/`: Uninhabited types

See [Uninhabited | Reference](https://doc.rust-lang.org/reference/glossary.html?highlight=Uninhabit#uninhabited).

## `tests/ui/union/`

See [Unions | Reference](https://doc.rust-lang.org/reference/items/unions.html).

## `tests/ui/unknown-unstable-lints/`: Attempting to refer to an unstable lint which does not exist

Tests for trying to use non-existent unstable lints.

**FIXME**: move this under `tests/ui/lints/`.

## `tests/ui/unop/`: Unary operators `-`, `*` and `!`

Tests the three unary operators for negating, dereferencing and inverting, across different contexts.

## `tests/ui/unpretty/`: `-Z unpretty` command line flag

The `-Z unpretty` flag outputs various representations of a program's tree in a certain way.

## `tests/ui/unresolved/`

Exercises various unresolved errors, ranging from earlier name resolution failures to later method resolution failures.

## `tests/ui/unsafe/`

A broad category of tests about unsafe Rust code.

## `tests/ui/unsafe-binders/`: `#![feature(unsafe_binders)]`

See [Tracking issue for unsafe binder types #130516](https://github.com/rust-lang/rust/issues/130516).

## `tests/ui/unsafe-fields/`: `struct`s and `enum`s with an `unsafe` field

See [Tracking issue for RFC 3458: Unsafe fields #132922](https://github.com/rust-lang/rust/issues/132922).

## `tests/ui/unsized/`: Zero-sized types, `Sized` trait, object has no known size at compile time

**FIXME**: between `tests/ui/zero-sized/`, `tests/ui/sized/` and this directory, might need to reorganize them a bit.

## `tests/ui/unsized-locals/`: `#![feature(unsized_locals, unsized_fn_params)]`

See:

- [RFC 1909 Unsized rvalues](https://github.com/rust-lang/rfcs/blob/master/text/1909-unsized-rvalues.md)
- [de-RFC 3829: Remove unsized_locals](https://github.com/rust-lang/rfcs/pull/3829)
- [Tracking issue for RFC #1909: Unsized Rvalues (`unsized_locals`, `unsized_fn_params`)](https://github.com/rust-lang/rust/issues/48055)

**FIXME**: Seems to also contain more generic tests that fit in `tests/ui/unsized/`.

## `tests/ui/unused-crate-deps/`

Exercises the `unused_crate_dependencies` lint.

## `tests/ui/unwind-abis/`

**FIXME**: Contains a single test, should likely be rehomed to `tests/ui/abi/`.

## `tests/ui/use/`

**FIXME**: merge with `ui/imports`.

## `tests/ui/variance/`: Covariants, invariants and contravariants

See [Variance | Reference](https://doc.rust-lang.org/reference/subtyping.html#variance).

## `tests/ui/variants/`: `enum` variants

Tests on `enum` variants.

**FIXME**: Should be rehomed with `tests/ui/enum/`.

## `tests/ui/version/`

**FIXME**: Contains a single test described as "Check that rustc accepts various version info flags.", should be rehomed.

## `tests/ui/warnings/`

**FIXME**: Contains a single test on non-explicit paths (`::one()`). Should be rehomed probably to `tests/ui/resolve/`.

## `tests/ui/wasm/`

These tests target the `wasm32` architecture specifically. They are usually regression tests for WASM-specific bugs which were observed in the past.

## `tests/ui/wf/`: Well-formedness checking

Tests on various well-formedness checks, e.g. [Type-checking normal functions](https://rustc-dev-guide.rust-lang.org/traits/lowering-to-logic.html).

## `tests/ui/where-clauses/`

Tests on `where` clauses. See [Where clauses | Reference](https://doc.rust-lang.org/reference/items/generics.html#where-clauses).

## `tests/ui/while/`

Tests on the `while` keyword and the `while` construct.

**FIXME**: merge with `ui/for-loop-while`.

## `tests/ui/windows-subsystem/`: `#![windows_subsystem = ""]`

See [the `windows_subsystem` attribute](https://doc.rust-lang.org/reference/runtime.html#the-windows_subsystem-attribute).

## `tests/ui/zero-sized/`: Zero-sized types

See [Zero-Sized Types | Reference](https://doc.rust-lang.org/nomicon/exotic-sizes.html#zero-sized-types-zsts).
