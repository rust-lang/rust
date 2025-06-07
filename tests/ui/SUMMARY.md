# UI Test Suite Categories

## `tests/ui/abi`: Application Binary Interface (ABI)

These tests deal with ABIs, mostly relating to function name mangling (and the `no_mangle` header), calling conventions, or compiler flags which affect ABI output and are passed in a compiletest header.

## `tests/ui/allocator`

These tests deal with features related to `allocator_api` or ̀`global_allocator`. If a test contains these, it probably belongs here.

[This tracking issue](https://github.com/rust-lang/rust/issues/32838) explains their function more in detail.

## `tests/ui/alloc-error`

These tests deal with features related to `alloc-error` and its handling. Information on this feature can be found in https://doc.rust-lang.org/std/alloc/fn.handle_alloc_error.html.

## `tests/ui/annotate-snippet`: Annotation snippets

These tests deal with `-error-format human-annotate-rs`, which allows for some customization of error messages with human-written annotations such as `//~? RAW hello to you, too!`

## `tests/ui/anon-params`: Anonymous Parameters

These tests deal with anonymous parameters, a deprecated feature. They should generate warnings in Rust 2015 edition, and cause hard errors in any later Rust edition. This category ensures this remains the case.

Using an anonymous parameter looks like this: `fn foo(i32)`, where the function parameter has no name - only its type is known.

## `tests/ui/argfile`: External files providing command line arguments

These tests get their command line arguments from a `.args` file and checks that this has the expected results.

If a test involves an `@argsfile` file and extracts command line parameters from it, it belongs in this category.

## `tests/ui/array-slice-vec`: Arrays, slices and vectors

The basic collection types `[]`, `&[]` and `Vec` involve numerous checks during compilation, such as type-checking, out-of-bounds indices, or even attempted instructions which are allowed in other programming languages (such as adding two vectors with the `+` operator).

These errors should be clear and helpful, and this category checks common cases and suitable error output.

## `tests/ui/argument-suggestions`: Argument suggestions

Calling a function with the wrong number of arguments causes a compilation failure, but the compiler is able to, in some cases, provide suggestions on how to fix the error, such as which arguments to add or delete.

This category checks that the suggestions are helpful and relevant to fixing the errors.

## `tests/ui/asm`: `asm!` macro

These tests deal with the `asm!` macro, which is used for adding inline assembly to Rust code. If a test contains this macro, it probably belongs here. Errors being tested revolve around using registers or operands incorrectly.

This directory contains subdirectories representing various architectures such as `riscv` or `aarch64`. If a test is specifically related to an architecture's particularities, it should be placed within the appropriate subdirectory - otherwise, architecture-independent tests may be placed below `tests/ui/asm` directly.

## `tests/ui/associated-consts`: Associated Constants

These tests deal with associated constants in traits and implementations. They verify correct behavior of constant definitions, usage, and type checking in associated contexts. For example, test that an associated trait constant will not segfault the compiler if it is expressed with a type nested 1000 times.

## `tests/ui/associated-inherent-types`: Inherent Associated Types

These tests cover associated types defined directly within inherent implementations (not in traits). For more information, [read the associated RFC](https://github.com/rust-lang/rfcs/blob/master/text/0195-associated-items.md#inherent-associated-items).

## `tests/ui/associated-item`: Associated Items

General tests for all kinds of associated items within traits and implementations. This category serves as a catch-all for tests that don't fit the other more specific associated item categories.

## `tests/ui/associated-type-bounds`: Associated Type Bounds

These tests verify the behavior of bounds on associated types in trait definitions and implementations, including `where` clauses that constrain associated types.

## `tests/ui/associated-types`: Trait Associated Types

Tests specifically focused on associated types within trait definitions and their implementations. If the associated type is not in a trait definition, it belongs in the `associated-inherent-types` category. This includes default associated types, overriding defaults, and type inference.

## `tests/ui/async-await`: Async/Await Syntax

Tests for the async/await syntax and related features. This includes tests for async functions, await expressions, and their interaction with other language features. If a test ensures error messages on futures and async/await programming are helpful, it belongs here.

## `tests/ui/attributes`: Compiler Attributes

Tests for compiler attributes and their behavior. This includes standard attributes like `#[derive]`, `#[cfg]`, and `#[repr]`, as well as their interactions and edge cases. For more information, [read the guide chapter on Attributes.](https://rustc-dev-guide.rust-lang.org/attributes.html)

## `tests/ui/auto-traits`: Auto Traits

There already are automatically derived traits (Send, Sync, etc.) but it is possible to make more with the unstable keyword ̀`auto`, which is [explained here](https://doc.rust-lang.org/beta/unstable-book/language-features/auto-traits.html). If a test uses this keyword, it belongs here.

## `tests/ui/autodiff`: Automatic Differentiation

The `#[autodiff]` macro handles automatic differentiation. This allows generating a new function to compute the derivative of a given function. It may only be applied to a function. [Read more here.](https://rustc-dev-guide.rust-lang.org/autodiff/internals.html)

Tests which use this feature belong here.

## `tests/ui/autoref-autoderef`: Automatic Referencing/Dereferencing

Tests for Rust's automatic referencing and dereferencing behavior, such as automatically adding reference operations (`&` or `&mut`) to make a value match a method's receiver type. Many of the tests here are `run-pass`.

## `tests/ui/auxiliary/`: Auxiliary support for tests directly under ̀`tests/ui`.

Many of these subdirectories contain an `auxiliary` subdirectory with files that do not test anything on their own, but provide support for a test. This top-level `auxiliary` subdirectory fulfills this role for the top-level `tests/ui` tests with no category.

NOTE: As these tests will eventually be rehomed or removed, this subdirectory will become obsolete.

## `tests/ui/backtrace/`: Backtrace Behaviour

Runtime panics and error handling generate backtraces to assist in debugging and diagnostics. These backtraces should be accurately generated and properly formatted, which is checked by these tests.

## `tests/ui/bench/`: Benchmarks and performance

This category was originally meant to contain tests related to time complexity and benchmarking.

However, only a single test was ever added to this category: https://github.com/rust-lang/rust/pull/32062

NOTE: It is also unclear what would happen were this test to "fail" - would it cause the test suite to remain stuck on this test for a much greater duration than normal?

## `tests/ui/binding/`: Pattern Binding

Tests for pattern binding behavior in match expressions, let statements, and other binding contexts. Includes tests for binding modes and refutability.

For more information, check https://doc.rust-lang.org/reference/patterns.html

## `tests/ui/binop/`: Binary operators

Tests for binary operator behavior (such as `==`, `&&` or `^`), including overloading, type checking, and error messages for invalid operations. 

## `tests/ui/blind/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/block-result/`: Block results and returning

Tests for block expression result behavior, including specifying the correct return types, semicolon handling, type inference, and expression/statement differentiation (for example, the difference between `1` and ̀`1;`).

If a test is intentionally playing with formatting to write blocks which appear correct at a glance to the human eye but not to the compiler, it belongs here.

## `tests/ui/bootstrap/`: RUSTC_BOOTSTRAP environment variable

NOTE: This category contains only a single test, which has its purpose described as follows:

```
//! Check the compiler's behavior when the perma-unstable env var `RUSTC_BOOTSTRAP` is set in the
//! environment in relation to feature stability and which channel rustc considers itself to be.
//!
//! `RUSTC_BOOTSTRAP` accepts:
//!
//! - `1`: cheat, allow usage of unstable features even if rustc thinks it is a stable compiler.
//! - `x,y,z`: comma-delimited list of crates.
//! - `-1`: force rustc to think it is a stable compiler.
```

## `tests/ui/borrowck/`: Borrow Checking

Tests for Rust's borrow checker, including tests for lifetime analysis, borrowing rules (such as borrowed data not escaping outside of a closure), and related error messages.

This is one of the biggest categories among these subdirectories, as this is one of Rust's most important language features.

## `tests/ui/box/`: Box Behavior

Tests for Rust's `Box<T>̀` smart pointer, including allocation behavior, dereference coercion, and edge cases in box pattern matching and placement. If a `Box` is the centerpiece of a test's success or failure, it belongs here.

## `tests/ui/btreemap/`: B-Tree Maps

Tests focused on `BTreeMap` collections and their compiler interactions. Includes tests for collection patterns, iterator behavior, and trait implementations specific to `BTreeMap`.

To read more on `BTreeMap`, see here: https://doc.rust-lang.org/std/collections/struct.BTreeMap.html

## `tests/ui/builtin-superkinds/`: Built-in Trait Hierarchy Tests

Tests for Rust's built-in trait hierarchy (Send, Sync, Sized, etc.) and their supertrait relationships. It checks compiler behavior regarding auto traits and marker trait constraints.

If you are definining your own auto traits with the `auto` keyword, try the `tests/ui/auto-traits` subdirectory instead.

## `tests/ui/cast/`: Type Casting

Tests for type casting behavior using the `as` operator. Includes tests for valid/invalid casts between primitive types, trait objects, and custom types. For example, check that trying to cast `i32` into `bool` results in a helpful error message.

## `tests/ui/cfg/`: Configuration Attribute

Tests for `#[cfg]` attribute behavior and conditional compilation. Checks handling of feature flags, target architectures, and other configuration predicates.

## `tests/ui/check-cfg/`: Configuration Verification

Tests for the `--check-cfg` compiler flag. It validates configuration names and values during compilation, and triggers errors when it detects mismatches with `#[cfg]̀` or `cfg!̀`.

## `tests/ui/closure_context/`: Closure type inference in context

In a closure, type inference is performed from surrounding scopes. Error messages about this should be clear, such as reporting that a closure should implement a certain trait but does not. These tests check the helpfulness of error messages in these cases.

## `tests/ui/closure-expected-type/`: Closure type inference

Some tests targeted at how we deduce the types of closure arguments. This process is a result of some heuristics which take into account the *expected type* we have alongside the *actual types* that we get from inputs.

NOTE: Appears to have significant overlap with `tests/ui/closure_context` and `tests/ui/functions-closures/closure-expected-type`. Needs further investigation.

## `tests/ui/closures/`: General Closure Tests

Any closure-focused tests that does not fit in the other closure categories belong here. Tests here involve syntax for writing closures, closures with the `move` keyword, or lifetimes not living long enough.

## `tests/ui/cmse-nonsecure/`: `C-cmse-nonsecure` ABIs

Tests for ̀`cmse_nonsecure_entry` and `abi_c_cmse_nonsecure_call`. Used specifically for the Armv8-M architecture, the former marks Secure functions with additional behaviours, such as adding a special symbol and constraining the number of parameters, while the latter alters function pointers to indicate they are non-secure and to handle them differently than usual.

For more information, consult:

- https://doc.rust-lang.org/beta/unstable-book/language-features/cmse-nonsecure-entry.html
- https://doc.rust-lang.org/beta/unstable-book/language-features/abi-c-cmse-nonsecure-call.html

## `tests/ui/codegen/`: Code Generation

Tests that check Rust's code generation. Tests revolving around codegen flags (starting with `-C` on the command line) are to be expected. Other potential topics include LLVM IR output, optimizations (and the various `opt-level`s), and target-specific code generation (such as tests specific to x86_64).

## `tests/ui/codemap_tests/`: Source Mapping

These tests intentionally try to break source code mapping with anomalies such as unusual spans and indentations, special characters (such as Chinese characters), and other ways of messing with how error messages try to report the exact line/column where an error occured.

## `tests/ui/coercion/`: Type Coercion

Tests for Rust's implicit type coercion behavior, where the types of some values are changed automatically depending on the context. This happens in cases such as automatic dereferencing or changing a `&mut` into a `&`.

Read https://doc.rust-lang.org/reference/type-coercions.html for more information.

## `tests/ui/coherence/`: Trait Implementation Coherence

Tests for Rust's trait coherence rules, which govern where trait implementations can be defined. Error reporting on orphan rule violations or trying to implement traits that overlap with pre-existing traits is tested in this subdirectory.

## `tests/ui/coinduction/`: Coinductive Trait Resolution

When rustc tries to resolve traits, it must sometimes assert that all its fields recursively implement an auto trait (such as `Send` or `Sync`), which sometimes results in infinite trees with cyclic trait dependencies. Coinduction circumvents this problem by proving the resolution of a trait despite its tree being possibly infinite.

These tests check the function of this coinduction feature.

For more information, read https://rustc-dev-guide.rust-lang.org/solve/coinduction.html

NOTE: This category only contains one highly specific test. Other coinduction tests can be found down the deeply located `tests/ui/traits/next-solver/cycles/coinduction` subdirectory. Some re-arrangements may be in order.

## `tests/ui/command/`: `std::process::Command`

Any test revolving around `std::process::Command` belongs here - **NOT** tests using `compile-flags` command-line instructions.

NOTE: the test ̀`command-line-diagnostics` seems to have been misplaced in this category.

## `tests/ui/compare-method/`: Trait implementation and definition comparisons

Some traits' implementation must be compared with their definition, checking for problems such as the implementation having stricter requirements (such as needing to implement `Copy`). The error messages of these cases should be clear and helpful.

This is **NOT** testing for comparison traits (`PartialEq`, `Eq`, ̀`PartialOrd̀`, `Ord`).

## `tests/ui/compiletest-self-test/`: Compiletest "meta" tests

These tests check the function of the UI test suite at large and Compiletest in itself. For example, check that the extra flags added by the header ̀`compile-flags` are added last and do not disrupt the order of default compilation flags.

## `tests/ui/conditional-compilation/`: Conditional Compilation

Tests for `#[cfg]` attribute or `--cfg` flags, used to compile certain files or code blocks only if certain conditions are met (such as developing on a specific architecture).

NOTE: There is significant overlap with `tests/ui/cfg`, which even contains a `tests/ui/cfg/conditional-compile.rs` test. Also investigate `tests/ui/check-cfg`.

## `tests/ui/confuse-field-and-method/`: Field/Method Ambiguity

If a developer tries to create a `struct̀` where one of the fields is a closure function, it becomes unclear whether `struct.field()` is accessing the field itself or trying to call the closure function within as a method. Error messages in this case should be helpful and correct the user's syntax.

## `tests/ui/const-generics/`: Constant Generics

Tests for const generics functionality, allowing types to be parameterized by constant values. It is generally observed in the form `<const N: Type>̀` after the `fn` or `struct` keywords. Includes tests for const expressions in generic contexts and associated type bounds.

For more information, see https://doc.rust-lang.org/reference/items/generics.html#const-generics

## `tests/ui/const_prop/`: Constant Propagation

If an expression already known at compile-time is used in Rust code (such as `[0, 1, 2, 3, 4, 5][3]`), the compiler will optimize it by replacing it with the corresponding, simpler value (`3` in this case). This is constant propagation. It has caused some internal compiler errors in the past, and these tests ensure these problems do not resurface.

See https://blog.rust-lang.org/inside-rust/2019/12/02/const-prop-on-by-default/

## `tests/ui/const-ptr/`: Constant Pointers

These tests heavily manipulate constant raw pointers, with operations involving arithmetic, casting and dereferencing, always with a `const`. This `unsafe` code has some pitfalls and require useful error messages, which are tested here.

## `tests/ui/consts/`: General Constant Evaluation

Anything to do with constants, which does not fit in the previous two `const` categories, goes here. This does not always imply use of the `const` keyword - other values considered constant, such as defining an enum variant as `enum Foo { Variant = 5 }` also counts.

## `tests/ui/contracts/`: Contracts feature

Any test revolving around `#![feature(contracts)]` fits here. Contracts are user-specified requirements for desired behaviour such as safety or correctness. For example, the following contract enforces that `x.baz` should always be superior to 0.

```rs
#[core::contracts::requires(x.baz > 0)]
fn doubler(x: Baz) -> Baz {
    Baz { baz: x.baz + 10 }
}
```

To read more about contracts, see here: https://github.com/rust-lang/compiler-team/issues/759

## `tests/ui/coroutine/`: Coroutines feature and `gen` blocks

If a test uses `#![feature(coroutines)]` or `gen` blocks, it belongs here. They have in common the usage of the `yield` keyword to yield values out of the coroutine or `gen` block, which can be used to implement custom iterators.

For coroutines, see here: https://doc.rust-lang.org/beta/unstable-book/language-features/coroutines.html

For more on `gen` blocks, read here: https://rust-lang.github.io/rfcs/3513-gen-blocks.html

## `tests/ui/coverage-attr/`: `#[coverage]` attribute

If a test uses `#![feature(coverage_attribute)]`, it belongs here. It uses `#[coverage]` to selectively disable coverage instrumentation in an annotated function. Read more here: https://github.com/rust-lang/rust/issues/84605

## `tests/ui/crate-loading/`: Crate Loading

Tests for crate resolution and loading behavior, including `extern crate` declarations, `--extern` flags, or the `use` keyword. These are some of the most common beginner mistakes in Rust, and require helpful error messages to assist new users.

## `tests/ui/cross/`: Various tests related to the concept of "cross"

NOTE: The unifying topic of these tests appears to be that their filenames begin with the word "cross". The similarities end there - one test is about "cross-borrowing" a `Box<T>` into `&T`, while another is about a global trait used "across" files.

## `tests/ui/cross-crate/`: Cross-Crate Interaction

Tests for behavior spanning multiple crates, including visibility rules, trait implementations, and type resolution across crate boundaries. `extern crate` lines are very common.

## `tests/ui/custom_test_frameworks/`: `#![feature(custom_test_frameworks)]`

Tests for alternative test harness implementations using `#![feature(custom_test_frameworks)]`. An example usage of this is specifying a test function accepting input (`test_runner`), then defining multiple `test_case`s to attempt different values with the `test_runner`-annotated function.

## `tests/ui/c-variadic/`: C Variadic Function

Tests for FFI functions with C-style variadic arguments (va_list). Uses the `extern` keyword extensively. An example test is trying to pass a function item to a variadic function, and the compiler signaling that a function pointer must be used instead.

## `tests/ui/cycle-trait/`: Trait Cycle Detection

Tests for detection and handling of cyclic trait dependencies. These unsolvable trait bounds cause infinite recursion in trait resolution, so it is important that the compiler finds the source of these cycles and warns the user about their location. An example of a cyclical trait:

```rs
trait A: B { }
trait B: C { }
trait C: B { }
```

## `tests/ui/dataflow_const_prop/`: Issue #131227

Contains a single test, described as follows:

```rs
//! Test that constant propagation in SwitchInt does not crash
//! when encountering a ptr-to-int transmute.
```

NOTE: A category with a single test again. Maybe it would fit inside the category `const-prop`.

## `tests/ui/debuginfo/`: Debug Information

Tests for generation of debug information (DWARF, etc.) including variable locations, type information, and source line mapping. There are many uses of flags such as `-C split-debuginfo` or `-C debuginfo`.

## `tests/ui/definition-reachable/`: Definition Reachability

Tests to check whether definitions (such as of a macro, placed inside an auxiliary file) are reachable from main execution paths (the test file). At the time of writing, most of the tests here are `run-pass`, meaning no error message is being checked.

## `tests/ui/delegation/`: `#![feature(fn_delegation)]`

Tests for the experimental delegation feature: syntactic sugar for delegating implementations of functions to other already implemented functions. For more information, read the RFC: https://github.com/rust-lang/rfcs/pull/3530

## `tests/ui/dep-graph/`: `-Z query-dep-graph`

These tests use the unstable command line option `query-dep-graph` to examine the dependency graph of a Rust program, which is useful for debugging. There are some pitfalls related to this option, such as `dump-dep-graph` requiring `query-dep-graph` to be enabled, and the related error messages are checked here.

## `tests/ui/deprecation/`: Deprecation Attribute

Tests for `#[deprecated]` or `deprecated_in_future` attribute behavior, as well as other errors and warnings about using deprecated Rust features.

## `tests/ui/deref-patterns/`: `#![feature(deref_patterns)]` and `#![feature(string_deref_patterns)]`

Tests for the `#![feature(deref_patterns)]` feature, which allows pattern matching on smart pointers in the standard library through their Deref target types, either implicitly or with explicit deref!(_) patterns. Read https://doc.rust-lang.org/nightly/unstable-book/language-features/deref-patterns.html for more information.

NOTE: May have some overlap with `tests/ui/pattern/deref-patterns`.

## `tests/ui/derived-errors/`: Derived Error Messages

In some cases, a very simple failure (such as not importing `HashMap`) would cause a cascade of other errors due to the type-checking problems this implies. The user would then be swarmed by error messages when the actual cause is hidden somewhere among them. The Rust compiler tries to avoid this and keep error messages minimal, with the derivatives of a main root cause silenced.

## `tests/ui/derives/`: Derive Macro

Tests for built-in derive macros (`Debug`, `Clone`, etc.) when used in conjunction with, for example, `#![derive(Copy)]`. Error messages should check that `derive` should be applied to the right types and check, for example, that all fields of a struct also implement the `derive` trait.

## `tests/ui/deriving/`: Derive Macro

NOTE: Appears to simply be a duplicate category of `tests/ui/derives`.

## `tests/ui/dest-prop/` Destination Propagation

NOTE: Contains a single test. The Rust compiler recognizes that the intermediate String values (from previous iterations) can be safely dropped or reused, avoiding unnecessary allocations. This optimization, enabled by `-Zmir-opt-level=3`, should still allow the code to `run-pass`. This could maybe be rehomed in a category about optimizations in general.

The test first appeared in https://github.com/rust-lang/rust/pull/72632.

## `tests/ui/destructuring-assignment/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/diagnostic-flags/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/diagnostic_namespace/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/diagnostic-width/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/did_you_mean/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/directory_ownership/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/disallowed-deconstructing/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/dollar-crate/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/drop/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/drop-bounds/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/dropck/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/dst/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/duplicate/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/dynamically-sized-types/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/dyn-compatibility/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/dyn-drop/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/dyn-keyword/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/dyn-star/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/editions/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/empty/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/entry-point/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/enum/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/enum-discriminant/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/env-macro/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/ergonomic-clones/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/error-codes/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/error-emitter/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/errors/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/explain/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/explicit/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/explicit-tail-calls/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/expr/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/extern/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/extern-flag/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/feature-gates/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/ffi-attrs/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/float/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/fmt/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/fn/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/fn-main/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/for/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/force-inlining/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/foreign/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/for-loop-while/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/frontmatter/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/fully-qualified-type/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/functional-struct-update/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/function-pointer/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/functions-closures/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/generic-associated-types/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/generic-const-items/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/generics/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/half-open-range-patterns/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/hashmap/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/hello_world/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/higher-ranked/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/hygiene/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/illegal-sized-bound/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/impl-header-lifetime-elision/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/implied-bounds/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/impl-trait/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/imports/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/include-macros/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/incoherent-inherent-impls/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/indexing/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/inference/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/infinite/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/inherent-impls-overlap-check/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/inline-const/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/instrument-coverage/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/instrument-xray/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/interior-mutability/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/internal/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/internal-lints/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/intrinsics/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/invalid/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/invalid-compile-flags/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/invalid-module-declaration/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/invalid-self-argument/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/io-checks/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/issues/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/iterators/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/json/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/keyword/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/kindck/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/label/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/lang-items/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/late-bound-lifetimes/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/layout/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/lazy-type-alias/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/lazy-type-alias-impl-trait/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/let-else/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/lexer/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/lifetimes/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/limits/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/linkage-attr/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/linking/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/link-native-libs/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/lint/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/liveness/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/loops/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/lowering/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/lto/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/lub-glb/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/macro_backtrace/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/macros/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/malformed/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/marker_trait_attr/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/match/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/meta/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/methods/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/mir/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/mir-dataflow`: Mid-level internal representation dataflow analysis

This directory contains unit tests for the MIR-based dataflow analysis.

These unit tests check the dataflow analysis by embedding calls to a special `rustc_peek` intrinsic within the code, in tandem with an attribute `#[rustc_mir(rustc_peek_maybe_init)]` (\*). With that attribute in place, `rustc_peek` calls are a signal to the compiler to lookup the computed dataflow state for the Lvalue corresponding to the argument expression being fed to `rustc_peek`. If the dataflow state for that Lvalue is a 1-bit at that point in the control flow, then no error is emitted by the compiler at that point; if it is a 0-bit, then that invocation of `rustc_peek` will emit an error with the message "rustc_peek: bit not set".

## `tests/ui/mismatched_types/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/missing/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/missing_non_modrs_mod/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/missing-trait-bounds/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/modules/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/modules_and_files_visibility/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/moves`: Passing ownership of values

These tests deal with moved values. If a piece of Rust code tries to have multiple variables take ownership of the same value, this code will fail compilation, and the various error messages this can generate should be tested in this category.

For more information about the most common errors tested in this category, try `rustc --explain E0382`.

## `tests/ui/mut/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/namespace/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/never_type/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/new-range/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/nll/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/non_modrs_mods/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/non_modrs_mods_and_inline_mods/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/no_std/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/not-panic/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/numbers-arithmetic/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/numeric/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/object-lifetime/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/obsolete-in-place/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/offset-of/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/on-unimplemented/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/operator-recovery/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/or-patterns/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/overloaded/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/packed/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/panic-handler/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/panic-runtime/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/panics/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/parallel-rustc/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/parser/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/patchable-function-entry/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/pattern/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/pin-macro/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/precondition-checks/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/print-request/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/print_type_sizes/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/privacy/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/process/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/process-termination/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/proc-macro/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/ptr_ops/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/pub/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/qualified/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/query-system/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/range/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/raw-ref-op/`: TODO(#141695)

TODO(#141695): add some description

## ̀`tests/ui/reachable`: reachable/unreachable code blocks

A variety of tests around reachability. These tests in general check two things:

- that we get unreachable code warnings in reasonable locations;
- that we permit coercions **into** `!` from expressions which
  diverge, where an expression "diverges" if it must execute some
  subexpression of type `!`, or it has type `!` itself.

## `tests/ui/recursion/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/recursion_limit/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/regions/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/repeat-expr/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/repr/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/reserved/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/resolve/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/return/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/rfcs/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/rmeta/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/runtime/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/rust-2018/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/rust-2021/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/rust-2024/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/rustc-env`: Rust compiler environment variables

Some environment variables affect rustc's behavior not because they are major compiler interfaces but rather because rustc is, ultimately, a Rust program, with debug logging, stack control, etc.

Use this category to group tests that use environment variables to control something about rustc's core UX, like "can we parse this number of parens if we raise RUST_MIN_STACK?" with related code for that compiler feature.

## `tests/ui/rustdoc`

This directory is for tests that have to do with rustdoc, but test the behavior of rustc. For example, rustc should not warn that an attribute rustdoc uses is unknown.

## `tests/ui/sanitizer/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/self/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/sepcomp/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/shadowed/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/shell-argfiles/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/simd/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/single-use-lifetime/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/sized/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/span/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/specializatioǹ`

This directory contains the test for incorrect usage of specialization that should lead to compile failure. Those tests break down into a few categories, such as feature gating, attempting to specialize without using `default` or attempting to change impl polarity in a specialization.

## `tests/ui/stability-attribute/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/stable-mir-print/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/stack-protector/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/static/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/statics/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/stats/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/std/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/stdlib-unit-tests/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/str/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/structs/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/structs-enums/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/suggestions/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/svh/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/symbol-mangling-version/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/symbol-names/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/sync/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/target-cpu/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/target-feature/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/target_modifiers/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/test-attrs/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/thir-print/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/thread-local/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/threads-sendsync/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/tool-attributes/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/track-diagnostics/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/trait-bounds/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/traits/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/transmutability/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/transmute/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/treat-err-as-bug/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/trivial-bounds/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/try-block/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/try-trait/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/tuple/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/type/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/type-alias/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/type-alias-enum-variants/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/type-alias-impl-trait/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/typeck/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/type-inference/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/typeof/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/ufcs/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/unboxed-closures/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/underscore-imports/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/underscore-lifetime/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/uniform-paths/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/uninhabited/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/union/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/unknown-unstable-lints/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/unop/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/unpretty/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/unresolved/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/unsafe/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/unsafe-binders/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/unsafe-fields/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/unsized/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/unsized-locals/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/unused-crate-deps/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/unwind-abis/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/use/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/variance/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/variants/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/version/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/warnings/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/wasm/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/wf/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/where-clauses/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/while/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/windows-subsystem/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/zero-sized/`: TODO(#141695)

TODO(#141695): add some description
