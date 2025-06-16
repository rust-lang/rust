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

## `tests/ui/destructuring-assignment/`

It is possible to destructure on assignment in the same way that it is possible in `let` declarations:

```rs
(a, (b.x.y, c)) = (0, (1, 2));
(x, y, .., z) = (1.0, 2.0, 3.0, 4.0, 5.0);
[_, f, *baz(), a[i]] = foo();
[g, _, h, ..] = ['a', 'w', 'e', 's', 'o', 'm', 'e', '!'];
Struct { x: a, y: b } = bar();
Struct { x, y } = Struct { x: 5, y: 6 };
```

These tests check the error messages related to this feature.

For more information, read the RFC: https://github.com/rust-lang/rfcs/blob/master/text/2909-destructuring-assignment.md

## `tests/ui/diagnostic-flags/`

These tests revolve around command-line flags which change the way error/warning diagnostics are emitted. For example, `--error-format=human --color=always`.

NOTE: Check redundancy with `annotate-snippet`.

## `tests/ui/diagnostic_namespace/`: `#[diagnostic]`

The `#[diagnostic]` attribute hints the compiler to emit specific diagnostic messages in certain cases like type mismatches, unsatisfied trait bounds or similar situations. These tests check the stability of this behaviour.

For more information, read the RFC: https://github.com/rust-lang/rfcs/blob/master/text/3368-diagnostic-attribute-namespace.md

## `tests/ui/diagnostic-width/`: `--diagnostic-width`

Everything to do with `--diagnostic-width`. This flag takes a number that specifies the width of the terminal in characters. Formatting of diagnostics will take the width into consideration to make them better fit on the screen.

## `tests/ui/did_you_mean/`: `help:` section in error messages

These tests are all about error messages caused by simple user typos, with a `help:` section to show a suggested correction. For example, writing `(0..13).collect<Vec<i32>>()` instead of `(0..13).collect::<Vec<i32>>()`.

## `tests/ui/directory_ownership/`: Declaring `mod` inside a block

These tests check the clarity of error messages when a code block attempts to gain ownership of a non-inline module with a `mod` keyword placed inside of it.

## `tests/ui/disallowed-deconstructing/`: Incorrect struct deconstruction

These tests check the clarity of error messages about struct deconstruction, and the various user errors that can be encountered in this pattern. For example, writing `let X { x: y } = x;` incorrectly instead of `let X { x: ref y } = x;`, where X implements the `Drop` trait.

## `tests/ui/dollar-crate/`: `$crate` used with the `use` keyword

There are a few rules - which are checked in this directory - to follow when using `$crate` - it must be used in the start of a `use` line and is a reserved identifier.

NOTE: There are a few other tests in other directories with a filename starting with `dollar-crate`. They should perhaps be redirected here.

## `tests/ui/drop/`: `Drop` and drop order

Not necessarily about `Drop` and its implementation, but also about the drop order of fields inside a struct. Tests are sometimes `run-pass`, checking that the order remains predictable.

## `tests/ui/drop-bounds/`: Bounding generic type on `Drop`

Bounding a generic type on ̀`Drop` is most likely a user error, so much that there is a default `warn`-level lint against it. These tests increase the lint level to `deny` to check the errors resulting from this lint.

## `tests/ui/dropck/`: Drop Checking

Mostly about checking the validity of `Drop` implementations, and the error messages which are printed in the event of an invalid implementation.

Additional reading:

- https://doc.rust-lang.org/nomicon/dropck.html
- https://rustc-dev-guide.rust-lang.org/borrow_check/drop_check.html

## `tests/ui/dst/`: Dynamically Sized Types

A type with a size that is known only at run-time is called a dynamically sized type (DST) or, informally, an unsized type. This has many implications in error messages, such as trying to assign a value into a DST.

For more information, read the reference: https://doc.rust-lang.org/reference/dynamically-sized-types.html

## `tests/ui/duplicate/`: Duplicate Symbols

Test about duplicated symbol names and associated errors, such as using the `export_name` attribute to rename a function with the same name as another function.

## `tests/ui/dynamically-sized-types/`: Dynamically Sized Types

NOTE: Duplicate category of `ui/dst`.

## `tests/ui/dyn-compatibility/`: `dyn`-compatible/Object Safety

Previously known as "object safety". Only some traits can be used with dynamic dispatch features using the `dyn` keyword - attempting to use non dyn-compatible traits in this fashion will result in the errors tested inside this directory.

## `tests/ui/dyn-drop/`: `dyn Drop`

NOTE: Contains a single test, used only to check the `dyn_drop` lint (which is normally `warǹ` level):

```
error: types that do not implement `Drop` can still have drop glue, consider instead using `std::mem::needs_drop` to detect whether a type is trivially dropped
```

## `tests/ui/dyn-keyword/`: `dyn`/Dynamic Dispatch

The `dyn` keyword is used to highlight that calls to methods on the associated Trait are dynamically dispatched. To use the trait this way, it must be dyn compatible - tests about dyn-compatibility belong in `tests/ui/dyn-compatibility/`, while more general tests on dynamic dispatch belong here.

For more information, read https://doc.rust-lang.org/std/keyword.dyn.html

## `tests/ui/dyn-star/`: `dyn*`/Sized `dyn`/`#![feature(dyn_star)]`

Stack-allocated parameters need to have a known size, and the compiler cannot know how big `dyn` is. `dyn*` is a sized version of `dyn` to solve this problem, added with `#![feature(dyn_star)]`.

For more information, read https://smallcultfollowing.com/babysteps//blog/2022/03/29/dyn-can-we-make-dyn-sized/

## `tests/ui/editions/`: Rust edition-specific peculiarities

These tests run in specific Rust editions, such as Rust 2015 or Rust 2018, and check errors and functionality related to specific now-deprecated idioms and features.

NOTE: Should be redistributed among `rust-2018`, `rust-2021` and `rust-2024`.

## `tests/ui/empty/`: Various tests related to the concept of "empty"

The tests in this category are quite eclectic. For example, one checks that `#[link(name = "")]` cannot be an empty string. Another checks that:

```rs
enum E {
    Empty3 {}
}

// SNIP

    match e3 {
        E::Empty3 => ()
        //~^ ERROR expected unit struct, unit variant or constant, found struct variant `E::Empty3`
    }
```

denies the `match` pattern, as the empty `{}` struct contents was not included in the pattern.

NOTE: This may be more excusable than `ui/cross`.

## `tests/ui/entry-point/`: `main` function

`main` is the entry-point of a Rust program - it must not be ambiguous or have generic parameters. These tests check the consistency of these errors and their messages.

## `tests/ui/enum/`: Enumerations/Tagged Unions

All general-purpose tests on the `enum` keyword, also known as tagged unions.

For more information, read the reference: https://doc.rust-lang.org/reference/items/enumerations.html

## `tests/ui/enum-discriminant/`: Discriminants

`enum` variants can be differentiated independently of their potential field contents with `discriminant`, which returns the type `Discriminant<T>`. Any test revolving around this feature belongs here.

For more information, read https://doc.rust-lang.org/std/mem/fn.discriminant.html

## `tests/ui/env-macro/`: `env!`

The `env!` macro is used to fetch environment variables, which is used in this test directory to `assert!` their returned values with expected defaults.

## `tests/ui/ergonomic-clones/`: ̀`#![feature(ergonomic_clones)]̀`

This feature simplifies performing lightweight clones (such as of `Arc`/`Rc`), particularly cloning them into closures or async blocks, while still keeping such cloning visible and explicit.

For more information, read the RFC: https://github.com/rust-lang/rfcs/pull/3680

## `tests/ui/error-codes/`: Error codes such as E0004

After a Rust error occurs, it is usually possible to open an explanatory page by running `rustc --explain EXXXX` where each X is a number. Error messages should be associated with their appropriate error code, which is tested within this directory.

## `tests/ui/error-emitter/`

Quite similar to `ui/diagnostic-flags` in some of its tests, this category checks some behaviours of Rust's error emitter into the user's terminal window, such as truncating error in the case of an excessive amount of them.

NOTE: Potentially check for overlap in this category, but this is not a priority.

## `tests/ui/errors/`: 

These tests are about very different topics, only unified by the fact that they result in errors.

NOTE: This should be cleaned up by sending each test where they belong, then deleted.

## `tests/ui/explain/`: `rustc --explain EXXXX`

A follow-up to `ui/error-codes/` - after the user sees an error code, they may try to read its explanation by entering this command-line argument. These tests check the functionality of this quality-of-life feature.

## `tests/ui/explicit/`: Errors involving the concept of "explicit"

This category contains three tests: two which are about the specific error `explicit use of destructor method`, and one which is about explicit annotation of lifetimes: https://doc.rust-lang.org/stable/rust-by-example/scope/lifetime/explicit.html.

NOTE: Rehome the two tests about the destructor method with `drop`-related categories, and rehome the last test with a category related to lifetimes.

## `tests/ui/explicit-tail-calls/`: `#![feature(explicit_tail_calls)]`

While tail call elimination (TCE) is already possible via tail call optimization (TCO) in Rust, there is no way to guarantee that a stack frame must be reused.

This feature provides tail call elimination via the `become` keyword, granting this guarantee.

For more information, read the RFC: https://github.com/rust-lang/rfcs/pull/3407

## `tests/ui/expr/`: Expressions

A broad category about Rust expressions. Most of the tests are within a subdirectory `ui/expr/if` about `if` conditionals specifically, while other tests are about errors containing the string `expected expression`.

## `tests/ui/extern/`: `extern` keyword

Tests about the `extern` keyword, such as in the usecase of extern function pointers.

## `tests/ui/extern-flag/`: `--extern` command line flag

Tests the `--extern` CLI flag, which, in the UI test suite, is called using `//@ aux-crate`.

## `tests/ui/feature-gates/`: `#![feature()]`

Many other test categories revolve around specific Rust features, but this category tests the feature gates that enable them. They should contain valid Rust feature names, and cause a compiler error if the user has not enabled unstable features.

## `tests/ui/ffi-attrs/`: `#![feature(ffi_const, ffi_pure)]`

The `#[ffi_const]` and `#[ffi_pure]` attributes applies clang's `const` and `pure` attributes to foreign functions declarations, respectively. These attributes are the core element of the tests in this category.

For more information, read:

- https://doc.rust-lang.org/beta/unstable-book/language-features/ffi-const.html
- https://doc.rust-lang.org/beta/unstable-book/language-features/ffi-pure.html

## `tests/ui/fmt/`: `format!` macro

These tests are all about the `format!` macro and its helpful error outputs (such as suggesting pretty-print with `{:?}`), as well as checking that situations which previously caused internal compiler errors do not do so again.

## `tests/ui/fn/`: Rust functions/`fn` keyword

A large, broad category encompassing errors and UI testing around Rust functions, such as checking the error message printed by having two functions with the same name.

## `tests/ui/fn-main/`: `main()` function

An extremely small category with 2 tests about the entry point of a Rust program.

NOTE: Serves a duplicate purpose with `ui/entry-point`, should be combined.

## `tests/ui/for/`: `for` keyword

Tests on the `for` keyword and some of its associated errors, such as attempting to write the faulty pattern `for _ in 0..1 {} else {}`.

NOTE: Should be merged with `ui/for-loop-while`.

## `tests/ui/force-inlining/`: `#[rustc_force_inline]`

Tests for `#[rustc_force_inline]`, which will force a function to always be labelled as inline by the compiler (it will be inserted at the point of its call instead of being used as a normal function call.) If the compiler is unable to inline the function, an error will be reported.

For more information, read the pull request: https://github.com/rust-lang/rust/pull/134082

## `tests/ui/foreign/`: Foreign Function Interface

Tests for `extern "C"` and `extern "Rust`, which allows Rust functions to integrate into C's application binary interface (ABI).

NOTE: Check for potential overlap/merge with `ui/c-variadic` and/or `ui/extern`.

## `tests/ui/for-loop-while/`: `for`, `loop` and `while` keywords

Anything to do with Rust loops and these three keywords to express them, with checks such as the error messages for trying to use `break` outside of a loop block.

NOTE: After `ui/for` is merged into this, also carry over its SUMMARY text.

## `tests/ui/frontmatter/`: `#![feature(frontmatter)]`

The `frontmatter` feature allows an extra metadata block at the top of files for consumption by external tools. It looks like this:

```rs
#!/usr/bin/env -S cargo -Zscript
---
[dependencies]
clap = "4"
---
```

For more information, read the RFC: https://github.com/rust-lang/rfcs/blob/master/text/3503-frontmatter.md

## `tests/ui/fully-qualified-type/`

A fully qualified type provides the maximum amount of information to prevent confusion. For example, if two modules `x` and `y` contain type `Foo`, the error message should report this type as `x::Foo` or `y::Foo` instead of the confusing `Foo` - which is tested in this directory.

## `tests/ui/functional-struct-update/`

Functional Struct Update is the name for the idiom by which one can write ..<expr> at the end of a struct literal expression to fill in all remaining fields of the struct literal by using <expr> as the source for them.

This is seen in some crates which will write patterns such as `let foo = Foo { x: 1, ..default() };`

For more information, check the RFC: https://github.com/rust-lang/rfcs/blob/master/text/0736-privacy-respecting-fru.md

## `tests/ui/function-pointer/`

Tests on function pointers, such as testing their compatibility with higher-ranked trait bounds (https://doc.rust-lang.org/nomicon/hrtb.html).

## `tests/ui/functions-closures/`

Tests on function closures, which do not use the `fn` keyword, but rather the `|| {}` structure.

For more information, read the Book chapter: https://doc.rust-lang.org/book/ch13-01-closures.html

## `tests/ui/generic-associated-types/`

Associated types (https://doc.rust-lang.org/rust-by-example/generics/assoc_items/types.html) allow for code readability by associating types with a trait, removing the need for functions using that trait to express the types that have been associated with the trait. This is extended by Generic Associated Types (GAT) allowing usage of generics within this pattern (https://blog.rust-lang.org/2022/10/28/gats-stabilization/).

## `tests/ui/generic-const-items/`: `#![feature(generic_const_items)]`

This feature allows generic parameters and where-clauses on free & associated const items, such as in the case of `const _IDENTITY<T>: fn(T) -> T = |x| x;`.

For more information, read this thread: https://github.com/rust-lang/lang-team/issues/214

## `tests/ui/generics/`

A large, broad category on the topic of generics (https://doc.rust-lang.org/rust-by-example/generics.html), with many tests on common syntax errors and preventing the resurgence of past internal compiler errors.

## `tests/ui/half-open-range-patterns/`: `x..` or `..x` range patterns

Tests on range patterns where one of the bounds is not a direct value. An example tests checks the error message returned from writing `x..=`, an inclusive open-ended range with no upper bound.

NOTE: Overlap with `ui/range`. `impossible_range.rs` is particularly suspected to be a duplicate test.

## `tests/ui/hashmap/`: `std::collections::HashMap`

Tests on the `std` implementation of `HashMap`, and its potential pitfalls with borrowing and collision.

## `tests/ui/hello_world/`: Extremely minimal "hello world" test

Tests that the basic `fn main() {println!("Hello, world!");}` program is not somehow broken.

## `tests/ui/higher-ranked/`: Higher-ranked trait bounds

It is sometimes necessary to create a trait bound which is valid for all lifetimes, where a Higher-Ranked Trait Bounds (HRTBs) will allow this. This test category checks errors and behaviours related to this feature.

For more information, read:

- https://rustc-dev-guide.rust-lang.org/traits/hrtb.html
- https://doc.rust-lang.org/nomicon/hrtb.html

## `tests/ui/hygiene/`: Various tests related to the concept of "cleanliness"

This seems to have been originally intended for "hygienic macros" - macros which work in all contexts, independent of what surrounds them. However, this category has grown into a mish-mash of many tests that may belong in the other categories of this list.

NOTE: Sort this category properly.

## `tests/ui/illegal-sized-bound/`: `Sized` trait objects illegal operations

This test category revolves around trait objects with `Sized` having illegal operations performed on them.

NOTE: There seems to be unrelated testing in this directory, such as `ui/illegal-sized-bound/mutability-mismatch-arg.rs`. Investigate.

## `tests/ui/impl-header-lifetime-elision/`: Lifetime elision in `impl` headers

In order to make common patterns more ergonomic, Rust allows lifetimes to be elided in function signatures. This is what allows one to write `fn print(s: &str);` instead of `fn print<'a>(s: &'a str);`. For impl headers, all types are input. So `impl Trait<&T> for Struct<&T>` has elided two lifetimes in input position, while `impl Struct<&T>` has elided one.

The tests in this directory revolve around this - for example, check that this feature is not yet supported for associated types, such as in `tests/ui/impl-header-lifetime-elision/assoc-type.rs`.

For more information, read https://doc.rust-lang.org/nomicon/lifetime-elision.html

## `tests/ui/implied-bounds/`: Implied lifetime bounds

Lifetimes which are essential (such as a function signature containing the type &'a T being only valid if T: 'a holds.) are sometimes inferred by the compiler. The tests in this directory check the consistency of the behaviour of this bound inference.

For more information, read https://doc.rust-lang.org/reference/trait-bounds.html#r-bound.implied

## `tests/ui/impl-trait/`: `impl` keyword for trait implementation

These tests revolve around the `impl` keyword's usage in traits, such as `impl TraitName for Foo` or `fn bar() -> impl TraitName`.

## `tests/ui/imports/`: `mod` and `use` keywords for imports

These tests check for errors related to wrong import paths and ambiguous names, sometimes from official Rust libraries, sometimes from user-defined auxiliary crates.

## `tests/ui/include-macros/`: `include!`, `include_str!`, `include_bytes!`

Tests the 3 macros `include!`, `include_str!` and `include_bytes!`, which are used to include content hosted in a separate file.

## `tests/ui/incoherent-inherent-impls/`: `cannot define inherent impl for a type outside of the crate where the type is defined`

Tests specifically for the error code `E0116`, obtainable when attempting to define an impl for a type defined outside the current crate.

## `tests/ui/indexing/`: Indices and out-of-bounds errors

Tests on collection types (arrays, slices, vectors) and various errors encountered when indexing their contents, such as accessing out-of-bounds values.

NOTE: (low-priority) could maybe be a subdirectory of `ui/array-slice-vec`

## `tests/ui/inference/`: Type inference/"type annotations needed"

Tests Rust's type inference and suggestions in the case of incorrect types, as well as the `type annotations needed` error visible in some scenarios, such as using `into()`.

## `tests/ui/infinite/`: "Recursive type `Foo` has infinite size"

Tests for the error message "Recursive type `Foo` has infinite size", which appears when creating types containing themselves (such as `struct Rec(Wrapper<Rec>);`)

## `tests/ui/inherent-impls-overlap-check/`: Repeating `impl` definitions across different blocks

This directory, which contains two tests, checks that repeating the same function names across separate `impl` blocks triggers an informative error, but not if the `impl` are for different types, such as `Bar<u8>` and `Bar<u16>`.

NOTE: This should maybe be a subdirectory within another related to duplicate definitions, such as `ui/duplicate`.

## `tests/ui/inline-const/`: `const` defined within function blocks

These tests revolve around the `const` keyword when it is used inside a `fn` block, which even allows it to have a return value such as `const { 1 + 7 }`.

## `tests/ui/instrument-coverage/`: `-Cinstrument-coverage` command line flag

When -C instrument-coverage is enabled, the Rust compiler enhances rust-based libraries and binaries by automatically injecting calls to an LLVM intrinsic, as well as embedding additional information in the data section of each library and binary.

For more information, read https://doc.rust-lang.org/rustc/instrument-coverage.html

## `tests/ui/instrument-xray/`: `-Z instrument-xray`

This command-line flag enables generation of NOP (no-operation) sleds for XRay function tracing instrumentation. XRay is a function call tracing system developed by LLVM.

## `tests/ui/interior-mutability/`

NOTE: Contains a single test. Its error message reads, `error[E0277]: the type UnsafeCell<i32> may contain interior mutability and a reference may not be safely transferrable across a catch_unwind boundary`. Perhaps this test could be relocated elsewhere.

## `tests/ui/internal/`: `extern crate internal_unstable;`

Tests for `internal_unstable` and the attribute header `#![feature(allow_internal_unstable)]`, which lets compiler developers mark features as internal to the compiler, and unstable for standard use.

## `tests/ui/internal-lints/`: rustc's lints for compiler contributors

While most lints are for end-users writing Rust programs, some lints are specifically tailored to suspicious patterns in compiler development. These lints' detection of these patterns are tested here.

## `tests/ui/intrinsics/`: `#![feature(intrinsics)]` and `#![feature(core_intrinsics)]`

Tests for the compiler's `std::intrinsics` module, meant to be used only in compiler development.

## `tests/ui/invalid/`: Various tests related to invalid input

These tests contains intentional errors such as providing an invalid value to a `#![crate_type="foo"]` header or the wrong path to a constant value (`u32::DOESNOTEXIST`).

NOTE: Possibly rehome into which directories directly concern each feature being made invalid.

## `tests/ui/invalid-compile-flags/`: `//@ compile-flags` invalidity

Attempts to pass erroneous flags using the `//@ compile-flags` compiletest directive. These are not necessarily nonexistent, but sometimes may only be used on a specific architecture.

## `tests/ui/invalid-module-declaration/`: `mod` referring to a nonexistent module

Contains a single test, which calls `mod baz` without `baz` existing anywhere.

NOTE: Consider merging with `ui/imports`.

## `tests/ui/invalid-self-argument/`: `self` as a function argument incorrectly

Tests with erroneous ways of using `self`, such as having it not be the first argument, or using it in a non-associated function (no `impl` or `trait`).

## `tests/ui/io-checks/`: File input/output

Contains a single test. The test tries to output a file into an invalid directory with `-o`, then checks that the result is an error, not an internal compiler error.

NOTE: Possibly rehome with a directory related to invalid command line flags, such as̀ ̀`ui/invalid-compile-flags/`̀̀

## `tests/ui/issues/`: Tests directly related to GitHub issues

These tests can concern almost any topic, their filename indicates which GitHub issue they refer to.

NOTE: This entire directory should be combed through, with its tests renamed and moved where they belong.

## `tests/ui/iterators/`

These tests revolve around anything to do with iteration. They feature, for example, trying to iterate over invalid types (such as `f32`), or trying to iterate over open-ended ranges where the first bound is not defined.

NOTE: Check for potential overlap with `ui/for-loop-while`.

## `tests/ui/json/`: `--json` command-line flag

These tests revolve around `--json`, used to output errors in the JSON format. It checks both correct usage of the flags and formatting of the output, as well as incorrect usage of `--json`.

## `tests/ui/keyword/`: Using Rust keywords out of their intended purpose

These tests use keywords as identifiers, such as `let true = 32`, where `true` is a reserved keyword, then check that these attempts are properly stopped by the compiler.

## `tests/ui/kindck/`: Kind check for sending between threads

These tests check whether certain types may be sent between threads using checks such as `assert_send`, returning error messages for invalid kinds of data being sent through.

## `tests/ui/label/`: Labelling blocks with lifetimes

These tests label code blocks with lifetimes (`'a {}`) in unusual ways, such as using restricted keywords for these label names (`'static`).

## `tests/ui/lang-items/`: `#![feature(lang_items)]`

The rustc compiler has certain pluggable operations, that is, functionality that isn't hard-coded into the language, but is implemented in libraries, with a special marker to tell the compiler it exists. The marker is the attribute `#[lang = "..."]` and there are various different values of ..., i.e. various different 'lang items'.

These tests check the behaviour of various 'lang items'.

For more information, read https://doc.rust-lang.org/nightly/unstable-book/language-features/lang-items.html

## `tests/ui/late-bound-lifetimes/`: rustc error code E0794

A lifetime parameter of a function definition is called late-bound if it both:

1.  appears in an argument type
2.  does not appear in a generic type constraint

This prevents defining the lifetime arguments explicitly, an error which is checked in various situations inside this directory.

## `tests/ui/layout/`: Type Layout

The layout of a type is its size, alignment, and the relative offsets of its fields. This has impact and potential errors when dealing with cases such as cyclical types or zero-sized types.

For more information, read: https://doc.rust-lang.org/reference/type-layout.html

## `tests/ui/lazy-type-alias/`: `#![feature(lazy_type_alias)]`

The feature implements the expected semantics for type aliases. Some of these include:

- Where-clauses and bounds on type parameters of type aliases are enforced.
- Type aliases are checked for well-formedness.

These tests check for expected behaviours of this feature.

## `tests/ui/lazy-type-alias-impl-trait/`: `#![feature(type_alias_impl_trait)]`

This feature allows use of an `impl Trait` in multiple locations while actually using the same concrete type (`type Alias = impl Trait;`) everywhere, keeping the original `impl Trait` hidden.

NOTE: Maybe rename this category to `ui/type_alias_impl_trait`.

## `tests/ui/let-else/`: `let Some(x) = Some(2) else {None};` syntax

Tests for the specific pattern which combines `let` and `else`, used for defining variables with an `Option` type.

## `tests/ui/lexer/`: Incorrect tokens

This test contains files which cause an error at the lexer stage, often due to invalid characters (such as using emoji in identifier names) or lacking spaces where there should be. 

## `tests/ui/lifetimes/`

A broader test category about lifetimes, including proper specifiers, lifetimes not living long enough, or undeclared lifetime names.

## `tests/ui/limits/`: Overly large values for a certain architecture

These tests feature gigantic values, such as `[[u8; 1518599999]; 1518600000]`, and check that the resulting errors point out the problem.

## `tests/ui/linkage-attr/`: `#![feature(linkage)]`

These tests make use of `#[linkage=""]`, where the empty string should be one of the various linkage models of LLVM. Some of the tests check for errors, such as specifying an invalid linkage model.

NOTE: Some of these tests do not use the feature at all, maybe move them to `ui/linking`.

## `tests/ui/linking/`: Various linker tests

Miscellaneous tests on code which fails during the linking stage, or which contain arguments and lines that have been known to cause unjustified errors in the past, such as specifying an unusual `export_name`.

## `tests/ui/link-native-libs/`: `#[link(name = "", kind = "")]` and `-l` command line flag

These tests accept special user-specified link arguments through either the attribute or the command line flag, and test that improper usage of these features results in the appropriate errors.

## `tests/ui/lint/`: Clippy lints

These tests will usually change the warn/deny level of Clippy lints, and check that code which should trigger various lints does, in fact, do so.

## `tests/ui/liveness/`: Dead code and moved variables

These tests check for unused variables, unreachable statements, functions which are supposed to return a value but do not, as well as values moved elsewhere before they could be used by a function.

NOTE: This seems unrelated to "liveness" as defined in the rustc compiler guide. Is this misleadingly named? https://rustc-dev-guide.rust-lang.org/borrow_check/region_inference/lifetime_parameters.html#liveness-and-universal-regions

## `tests/ui/loops/`: `loop` keyword

Tests on the `loop` keyword and some of its associated errors, such as attempting to write the faulty pattern `loop {} else {}`.

NOTE: Consider merging with `ui/for-loop-while`.

## `tests/ui/lowering/`: AST Lowering

The AST lowering step converts AST to HIR. This means many structures are removed if they are irrelevant for type analysis or similar syntax agnostic analyses. For example, ̀`if let` becomes ̀`match`.

These tests check edge cases which have been known to interact with AST lowering.

For more information, read https://rustc-dev-guide.rust-lang.org/ast-lowering.html

## `tests/ui/lto/`: Link Time Optimization flags such as `-C lto` or ̀`-Z thinlto`

These tests check LLVM's link time optimization feature, which makes some assumptions when linking code. These assumptions have caused problems when combining certain command-line flags or writing unusual patterns, which are checked inside this directory.

## `tests/ui/lub-glb/`: LUB/GLB algorithm update

Pull request [#45853](https://github.com/rust-lang/rust/pull/45853) changed the way the LUB/GLB algorithm is implemented, which has some breaking changes for certain functions which previously compiled. These old functions, as well as their fixed modern counterparts, are tested here.

For more information, read https://github.com/rust-lang/rust/issues/45852

## `tests/ui/macro_backtrace/`: `-Zmacro-backtrace`

Contains a single test, checking the unstable command-line flag to enable detailed macro backtraces.

NOTE: This could be merged with `ui/macros`, which already contains other macro backtrace tests.

## `tests/ui/macros/`

This is a broad category on Rust macros, including the errors which concern them, as well as the default `std` macros, such as using `panic!` inside a statement or expression.

## `tests/ui/malformed/`: Syntax errors in attributes

These tests are for simple typos in attributes, such as writing `#[allow { foo_lint } ]` instead of `#[allow ( foo_lint ) ]`.

## `tests/ui/marker_trait_attr/`: `#![feature(marker_trait_attr)]`

Normally, Rust keeps you from adding trait implementations that could overlap with each other, as it would be ambiguous which to use. This feature, however, carves out an exception to that rule: a trait can opt-in to having overlapping implementations, at the cost that those implementations are not allowed to override anything.

These tests use the `#[marker]` attribute for this purpose.

For more information, read https://doc.rust-lang.org/nightly/unstable-book/language-features/marker-trait-attr.html

## `tests/ui/match/`: `match` keyword

A broad category about `match` pattern matching, including some cases where miscompilation has previously been observed, or simpler error message checking for common mistakes such as non-exhaustive coverage.

## `tests/ui/meta/`: Tests for compiletest itself

These tests check the function of the UI test suite at large and Compiletest in itself.

NOTE: This should absolutely be merged with `tests/ui/compiletest-self-test/`.

## `tests/ui/methods/`

A broad category for anything related to methods (such as `foo.bar()`). Checks non-existent method calls, wrong number of arguments or attempts to mutate immutables.

## `tests/ui/mir/`: Failures and problems at the MIR step of `rustc`

Mid-level representation contains optimizations - and therefore, assumptions - which has caused problems with certain edge cases and unusual patterns. Cases which have previously caused issues in the MIR step are tested in this directory.

## `tests/ui/mir-dataflow`: Mid-level internal representation dataflow analysis

This directory contains unit tests for the MIR-based dataflow analysis.

These unit tests check the dataflow analysis by embedding calls to a special `rustc_peek` intrinsic within the code, in tandem with an attribute `#[rustc_mir(rustc_peek_maybe_init)]` (\*). With that attribute in place, `rustc_peek` calls are a signal to the compiler to lookup the computed dataflow state for the Lvalue corresponding to the argument expression being fed to `rustc_peek`. If the dataflow state for that Lvalue is a 1-bit at that point in the control flow, then no error is emitted by the compiler at that point; if it is a 0-bit, then that invocation of `rustc_peek` will emit an error with the message "rustc_peek: bit not set".

For more information, read: https://rustc-dev-guide.rust-lang.org/mir/dataflow.html

## `tests/ui/mismatched_types/`: Type mismatch errors, such as `error[E0308]` or `error[E0631]`

These tests check for cases where Rust's type system expects a certain type, but receives another, such as `let x: u32 = ()`.

## `tests/ui/missing/`: Tests which could pass if something extra was added

In these tests, an error is printed, and the code could compile if something was added, such as a `return` statement or a trait implementation (e.g.: ̀`Debug`).

NOTE: (low-priority) This is a massively broad category, and some of these tests could be rehomed. For example, `ui/return`.

## `tests/ui/missing_non_modrs_mod/`: `mod` tree with missing file at root

This directory is a small tree of `mod` dependencies, but the root, `foo.rs`, is looking for a file which does not exist. The test checks that the error is reported at the top-level module.

NOTE: Merge with `tests/ui/invalid-module-declaration/` or `tests/ui/imports`.

## `tests/ui/missing-trait-bounds/`: Type parameters needing additional restriction

In these tests, an operation is recognized to exist (such as calling `.clone()`, but a trait bound is missing (such as `Clonè`)). The errors are tested to report both the missing trait bound, and a suggestion on how to add it.

## `tests/ui/modules/`: `mod` and `use` keywords for imports

These tests check for errors related to invalid imports from user-defined auxiliary crates, including peculiar usages such as writing a macro to insert `mod` and `pub use` automatically.

NOTE: Merge with `ui/imports`.

## `tests/ui/modules_and_files_visibility/`: Using non-existent or inaccessible functions from modules

This small directory uses `mod` to import a crate, but calls on its functions incorrectly, using names which do not exist or are not visible.

NOTE: Merge with `ui/imports`.

## `tests/ui/moves`: Passing ownership of values

These tests deal with moved values. If a piece of Rust code tries to have multiple variables take ownership of the same value, this code will fail compilation, and the various error messages this can generate should be tested in this category.

For more information about the most common errors tested in this category, try `rustc --explain E0382`.

## `tests/ui/mut/`: Mutability

A broad category on mutability, such as the `mut` keyword, borrowing a value as both immutable and mutable (and the associated error), or adding mutable references to `const` declarations.

## `tests/ui/namespace/`

Contains a single test. It imports a massive amount of very similar types from a crate, then attempts various permutations of their namespace paths, checking for errors or the lackthereof.

NOTE: This should definitely be rehomed, maybe in `ui/imports`?

## `tests/ui/never_type/`: `#![feature(never_type)]`

Allows the `!` type, an empty type. An empty type is a type with no inhabitants, ie. a type for which there is nothing of that type, such as `enum Never {}`.

For more information, read the RFC: https://github.com/rust-lang/rfcs/blob/master/text/1216-bang-type.md

## `tests/ui/new-range/`: `#![feature(new_range)]`

Changes the range operators `a..b`, `a..`, and `a..=b` to resolve to new types `std::range::Range`, `std::range::RangeFrom`, and `std::range::RangeInclusive` in Edition 2024.

For more information, read the RFC: https://github.com/rust-lang/rfcs/blob/master/text/3550-new-range.md

## `tests/ui/nll/`: Non-lexical lifetimes

This fully stabilized feature extends Rust’s borrow system to support non-lexical lifetimes – these are lifetimes that are based on the control-flow graph, rather than lexical scopes.

For more information, read the RFC: https://rust-lang.github.io/rfcs/2094-nll.html

## `tests/ui/non_modrs_mods/`: Sprawling module tree

Despite the size of the directory, this is a single test, spawning a sprawling `mod` dependency tree and checking its successful build.

NOTE: Consider merge with `ui/imports`, keeping the directory.

## `tests/ui/non_modrs_mods_and_inline_mods/`: TODO(#141695)

A very similar principle as `non_modrs_mods`, but with an added inline `mod` statement inside another `mod`'s code block.

NOTE: Consider merge with `ui/imports`, keeping the directory.

## `tests/ui/no_std/`: `#![no_std]`

Tests where the standard library is disabled, such as checking that trying to unwind panics in a `no_std` environment does not result in weird messages about internal lang items.

## `tests/ui/not-panic/`: Types which are not safe to unwind

Older tests checking various types, such as `&RefCell<i32>`, and whether they are not `UnwindSafe` as expected.

## `tests/ui/numbers-arithmetic/`: Edge cases in integer and float mathematics

These tests attempt edge cases, such as specific floats, large or very small numbers, or bit conversion, and check that the arithmetic results are as expected.

## `tests/ui/numeric/`: Number types, such as `u8` or `isize`

These tests check number types and their interactions, such as casting among them with `as` or providing the wrong numeric suffix.

## `tests/ui/object-lifetime/`

Tests on lifetimes' effects on objects, such as a lifetime bound not being able to be deduced from context, or checking that lifetimes are inherited properly.

NOTE: Just a more specific subset of `ui/lifetimes`.

## `tests/ui/obsolete-in-place/`: Obsolete `x <- y` syntax

Contains a single test. Attempts to use the ancient Rust syntax `x <- y` and checks for its failure.

NOTE: Definitely should be rehomed, maybe to `ui/deprecation`.

## `tests/ui/offset-of/`: `offset_of!` macro

This macro expands to the offset in bytes of a field from the beginning of the given type. These tests use this feature in various ways, such as giving it invalid input and checking the resulting error.

For more information, read: https://doc.rust-lang.org/beta/std/mem/macro.offset_of.html

## `tests/ui/on-unimplemented/`: `rustc_on_unimplemented` and "X does not implement Y" errors

The` ̀#[rustc_on_unimplemented]̀` attribute allows trait definitions to add specialized notes to error messages when an implementation was expected but not found.

## `tests/ui/operator-recovery/`: Attempting to use `<=>` or `<>`

These tests use the invalid comparator syntax `<=>` or `<>`, then check for the corresponding error.

NOTE: Contains only 2 tests. Consider merge with `ui/did_you_mean`.

## `tests/ui/or-patterns/`: `|` and `||` as logical OR

These tests check OR operators, such as `let x = 5 || 6;` or `Foo::Bar | Foo::Baz` in match patterns.

## `tests/ui/overloaded/`: Operator Overloading

Operator overloading allows for some operators to accomplish different tasks based on their input arguments. This means that different results will be obtained from the same method, if it is used on different inputs which all implement the trait the method is related to.

For more information, read: https://doc.rust-lang.org/rust-by-example/trait/ops.html

## `tests/ui/packed/`: `#[repr(packed)]`

`repr(packed(n))` (where n is a power of two) forces the type to have an alignment of at most n. Most commonly used without an explicit n, `repr(packed)` is equivalent to `repr(packed(1))` which forces Rust to strip any padding, and only align the type to a byte. This may improve the memory footprint, but will likely have other negative side-effects.

For more information, read: https://doc.rust-lang.org/nomicon/other-reprs.html#reprpacked-reprpackedn

## `tests/ui/panic-handler/`: `#[panic_handler]` attribute

`#[panic_handler]` is used to define the behavior of ̀`panic!̀` in `#![no_std]` applications. Not all tests here are `no_std`, some check what happens if `panic_handler` is used in a `std` environment.

For more information, read: https://doc.rust-lang.org/nomicon/panic-handler.html

## `tests/ui/panic-runtime/`: `#![panic_runtime]`, `-Cpanic=` command line flag & panic unwinding strategy

These tests are about changing the panic-on-runtime strategy, as in, how the compiler will handle panics should they occur and whether it will try to unwind the panic, or abort immediately.

For more information, read the RFC: https://github.com/rust-lang/rfcs/blob/master/text/1513-less-unwinding.md

## `tests/ui/panics/`

An extremely broad category about panics in general, often but not necessarily using the `panic!` macro.

## `tests/ui/parallel-rustc/`: `-Zthreads=` command line flag

Runs rustc on a defined number of threads specified by the unstable command line flag. Checks for expected error messages or successful runs.

## `tests/ui/parser/`: Erroneous usage of keywords and failures at the parser stage

These tests use keywords and symbols improperly, such as writing `unsafe mod` or `mod break`. The errors are checked to be helpful and provide suggestions.

NOTE: This has some overlap with `ui/keyword`, and some of its tests should perhaps be moved.

## `tests/ui/patchable-function-entry/`: ̀`#![feature(patchable_function_entry)]̀`

The `-Z patchable-function-entry=total_nops,prefix_nops` or `-Z patchable-function-entry=total_nops` compiler flag enables nop padding of function entries with 'total_nops' nops, with an offset for the entry of the function at 'prefix_nops' nops.

For more information, read: https://doc.rust-lang.org/nightly/unstable-book/compiler-flags/patchable-function-entry.html

## `tests/ui/pattern/`

These tests revolve around patterns, which are used to match values against structures. This is not necessarily with the `match` keyword, as they can be observed in other contexts, such as `if let`.

For more information, read https://doc.rust-lang.org/reference/patterns.html

NOTE: Potentially a lot of overlap with `ui/match`.

## `tests/ui/pin-macro/`: `pin!`

This macro is a quick way of building `Pin` pointers, which prevents the value referenced by that pointer from being moved or otherwise invalidated at that place in memory. Attempting to do one of these things will result in the errors checked within this directory.

## `tests/ui/precondition-checks/`: `unsafe` features failing to meet preconditions

When using `unsafe` features such as `ptr::read` or `ptr::write`, the compiler still performs a degree of checking for preconditions that ensures a base, but incomplete level of safety for these features. Violating these preconditions results in the errors checked within this directory.

## `tests/ui/print-request/`: `--print` command line flag

These tests will request information from the compiler with `--print`, then compare the output with a sample `stdout` file to check for expected output.

For more information, read: https://doc.rust-lang.org/rustc/command-line-arguments/print-options.html

## `tests/ui/print_type_sizes/`: `-Z print-type-sizes` command line flag

An unstable flag to print out the size of each type in a Rust program. In this testing directory, the output is compared with an expected `stdout` file.

Output lines look like this: `print-type-size type: std::mem::ManuallyDrop<[u8; 8192]>: 8192 bytes, alignment: 1 bytes`

## `tests/ui/privacy/`: `pub` keyword & visibility

A large category about function and type public/private visibility, and its impact when using features across crates. Checks both visibility-related error messages and previously buggy cases.

## `tests/ui/process/`: Interacting with the operating system within Rust code

While these tests most commonly use `std::process::Command`, some refrain from importing it, instead interacting with environment variables.

NOTE: Potential overlap with `ui/command`.

## `tests/ui/process-termination/`: Exiting the main thread

These tests end execution of the program manually, one with `std::process::exit` from within a thread and another by iterating through 3 threads which are blocked on I/O, yielding each one then exiting the main thread normally. In both cases, the process should successfully terminate.

## `tests/ui/proc-macro/`: Procedural macros

A broad category on proc-macros: Rust functions which consume and produce Rust syntax, usually in the form of a `TokenStream`.

For more information, read: https://doc.rust-lang.org/reference/procedural-macros.html

## `tests/ui/ptr_ops/`: Using operations on a pointer

Contains only 2 tests, related to a single issue, which was about an error caused by using addition on a pointer to `i8`.

NOTE: This should likely be rehomed somewhere, but exactly where is unclear.

## `tests/ui/pub/`: `pub` keyword

A large category about function and type public/private visibility, and its impact when using features across crates. Checks both visibility-related error messages and previously buggy cases.

NOTE: Basically the same thing as `ui/privacy`, should be merged.

## `tests/ui/qualified/`: Qualified paths

Contains few tests on qualified paths where a type parameter is provided at the end: `type A = <S as Tr>::A::f<u8>;`. The tests check if this fails during type checking, not parsing.

NOTE: Should be rehomed to `ui/typeck`.

## `tests/ui/query-system/`

Tests on Rust methods and functions which use the query system, such as `std::mem::size_of`. These compute information about the current runtime and return it.

For more information, read: https://rustc-dev-guide.rust-lang.org/query.html

## `tests/ui/range/`: `0..1`-type ranges and explicit `Range`-related types

A broad testing category for Rust ranges, both in their `..` or `..=` form, as well as the `Range`, `RangeTo`, `RangeFrom` or `RangeBounds` types.

NOTE: May have some duplicate tests with `ui/new-range`.

## `tests/ui/raw-ref-op/`: Using operators on `&raw` values

`&raw mut <place>`, which creates a `*mut <T>`, and `&raw const <place>`, which creates a `*const <T>` allow for direct creation of a raw pointer. The tests in this directory use operators on these raw pointers, checking sometimes for `run-pass`, sometimes for compilation errors.

For more information, read the RFC: https://github.com/rust-lang/rfcs/blob/master/text/2582-raw-reference-mir-operator.md

## ̀`tests/ui/reachable`: reachable/unreachable code blocks

A variety of tests around reachability. These tests in general check two things:

- that we get unreachable code warnings in reasonable locations;
- that we permit coercions **into** `!` from expressions which
  diverge, where an expression "diverges" if it must execute some
  subexpression of type `!`, or it has type `!` itself.

NOTE: Check for overlap with `ui/liveness`.

## `tests/ui/recursion/`: Recursion and `#![recursion_limit = ""]`

Tests for recursion in Rust, not just in functions, but also in macros, `type` definitions, and more. Sometimes sets a `recursion_limit` with checks on how this affects compilation.

## `tests/ui/recursion_limit/`: `#![recursion_limit = ""]`

Sets a recursion limit on recursive code, then checks if this attribute was used correctly, and, if yes, how it affects compilation.

NOTE: Should be merged with `ui/recursion`, or at least have the `ui/recursion` tests about this attribute moved here.

## `tests/ui/regions/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/repeat-expr/`: `[Type; n]` syntax for creating arrays with repeated types across a set size

Tests the `[Type; 5]` syntax, which would create `̀[Type, Type, Type, Type, Type]`. Checks for various scenarios, such as using this syntax in statics (should work) or using a comma instead of a semicolon (should not, and the error message should be helpful).

NOTE: Maybe make this a subdirectory of `ui/array-slice-vec`.

## `tests/ui/repr/`: `#[repr(_)]`

Tests on `repr`, which lets the user define the alignment of a given type. For example, `repr(C)` uses the alignment of the C programming language.

NOTE: Maybe make this a subdirectory of `ui/attributes`.

## `tests/ui/reserved/`: Usage of keywords in attribute names and identifiers

Uses `rustc` in an attribute name and `become` as an identifier, which are reserved keywords, then tests for compilation failure and helpful suggestion.

NOTE: Should be merged with `ui/keyword`.

## `tests/ui/resolve/`: Name resolution

Tests on the name resolution phase of compilation. This is the phase which allows catching errors such as writing `Foo::Bar(32)` when `Foo::Bar{ x: 32 }` was expected.

For more information, read: https://rustc-dev-guide.rust-lang.org/name-resolution.html

## `tests/ui/return/`: `return` keyword

Tests on the `return` keyword, such as wrongfully attempting to use it outside of a function body.

## `tests/ui/rfcs/`: Tests related to a specific RFC

Tests tied to a Rust request for change (RFC). These cover any topic contained within the pages of https://rust-lang.github.io/rfcs/.

NOTE: Check if any of these is related to an unstable feature which already has its own directory.

## `tests/ui/rmeta/`: `--emit=metadata` command line flag

These tests use a crate which emits the metadata type, and checks for various problems - for example, using it as a dependency of an `rlib` crate, or trying to build a ̀`metadata` crate with an unspecified type.

## `tests/ui/runtime/`: TODO(#141695)

TODO(#141695): add some description

## `tests/ui/rust-2018/`

## `tests/ui/rust-2021/`

## `tests/ui/rust-2024/`

## Tests specific to a certain `rustc` edition

These tests often revolve around attempting to use an obsolete feature or keyword in a modern edition, or important observed bugs which only existed in a specific edition.

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

## `tests/ui/variance/`: Covariants, invariants and contravariants

Variance is a property that generic types have with respect to their arguments. A generic type’s variance in a parameter is how the subtyping of the parameter affects the subtyping of the type.

An example test in this directory checks that bounds on type parameters (other than `Self`) do not influence variance.

For more information, read: https://doc.rust-lang.org/reference/subtyping.html#variance

## `tests/ui/variants/`: `enum` variants

Tests on the `enum` keyword's variants, such as the error message when they are used as types without the preceding enum type in the path.

NOTE: Should be rehomed with `ui/enum`.

## `tests/ui/version/`: Single issue test on `--version`

NOTE: Contains a single test described as "Check that rustc accepts various version info flags.". Should be rehomed into a new directory about command-line flag tests in general.

## `tests/ui/warnings/`: Single issue test

NOTE: Contains a single test on non-explicit paths (`::one()`). Should be rehomed, but the location is unclear.

## `tests/ui/wasm/`: `//@ only-wasm32`

These tests target the `wasm32` architecture specifically. They are usually regression tests for WASM-specific bugs which were observed in the past.

## `tests/ui/wf/`: Well-formedness checking

For each declaration in a Rust program, we will generate a logical goal and try to prove it using [lowered rules](https://rust-lang.github.io/chalk/book/clauses/lowering_rules.html). If we are able to prove it, we say that the construct is well-formed. If not, we report an error to the user.

These tests check the expected behaviour of these proofs.

For more information, read: https://rust-lang.github.io/chalk/book/clauses/wf.html

## `tests/ui/where-clauses/`: `where` keyword

Tests on `where`, usually used to define the trait implementations of a generic parameter. For example, test that we can quantify lifetimes outside a constraint inside a `where` clause.

## `tests/ui/while/`: `while` keyword

Usage of the `while` keyword both correctly and incorrectly. An example of the latter: `while {} else {}`.

NOTE: Merge with `ui/for-loop-while`.

## `tests/ui/windows-subsystem/`: `#![windows_subsystem = ""]`

The `windows_subsystem` attribute may be applied at the crate level to set the subsystem when linking on a Windows target. Tests in this directory attempt to use this attribute correctly and incorrectly, checking for expected results.

For more information, read: https://doc.rust-lang.org/reference/runtime.html#the-windows_subsystem-attribute

## `tests/ui/zero-sized/`: Zero-sized types

Tests on zero-sized types, such as `struct Zero;`. Checks their interactions with, among others, linked lists and destructors.

For more information, read: https://doc.rust-lang.org/nomicon/exotic-sizes.html#zero-sized-types-zsts
