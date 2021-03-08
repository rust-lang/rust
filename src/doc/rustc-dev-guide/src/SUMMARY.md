# Summary

[About this guide](./about-this-guide.md)

[Getting Started](./getting-started.md)

---

# Building and debugging `rustc`

- [How to Build and Run the Compiler](./building/how-to-build-and-run.md)
    - [Prerequisites](./building/prerequisites.md)
    - [Suggested Workflows](./building/suggested.md)
    - [Distribution artifacts](./building/build-install-distribution-artifacts.md)
    - [Documenting Compiler](./building/compiler-documenting.md)
    - [Rustdoc overview](./rustdoc.md)
    - [ctags](./building/ctags.md)
    - [Adding a new target](./building/new-target.md)
- [The compiler testing framework](./tests/intro.md)
    - [Running tests](./tests/running.md)
    - [Adding new tests](./tests/adding.md)
    - [Using `compiletest` commands to control test execution](./compiletest.md)
- [Debugging the Compiler](./compiler-debugging.md)
- [Profiling the compiler](./profiling.md)
    - [with the linux perf tool](./profiling/with_perf.md)
    - [with Windows Performance Analyzer](./profiling/wpa_profiling.md)
- [crates.io Dependencies](./crates-io.md)


# Contributing to Rust

- [Introduction](./contributing.md)
- [About the compiler team](./compiler-team.md)
- [Using Git](./git.md)
- [Mastering @rustbot](./rustbot.md)
- [Walkthrough: a typical contribution](./walkthrough.md)
- [Bug Fix Procedure](./bug-fix-procedure.md)
- [Implementing new features](./implementing_new_features.md)
- [Stability attributes](./stability.md)
- [Stabilizing Features](./stabilization_guide.md)
- [Feature Gates](./feature-gates.md)
- [Coding conventions](./conventions.md)
- [Notification groups](notification-groups/about.md)
    - [ARM](notification-groups/arm.md)
    - [Cleanup Crew](notification-groups/cleanup-crew.md)
    - [LLVM](notification-groups/llvm.md)
    - [RISC-V](notification-groups/risc-v.md)
    - [Windows](notification-groups/windows.md)
- [Licenses](./licenses.md)

# High-level Compiler Architecture

- [Prologue](./part-2-intro.md)
- [Overview of the Compiler](./overview.md)
- [The compiler source code](./compiler-src.md)
- [Bootstrapping](./building/bootstrapping.md)
- [Queries: demand-driven compilation](./query.md)
    - [The Query Evaluation Model in Detail](./queries/query-evaluation-model-in-detail.md)
    - [Incremental compilation](./queries/incremental-compilation.md)
    - [Incremental compilation In Detail](./queries/incremental-compilation-in-detail.md)
    - [Debugging and Testing](./incrcomp-debugging.md)
    - [Profiling Queries](./queries/profiling.md)
    - [Salsa](./salsa.md)
- [Memory Management in Rustc](./memory.md)
- [Serialization in Rustc](./serialization.md)
- [Parallel Compilation](./parallel-rustc.md)
- [Rustdoc internals](./rustdoc-internals.md)

# Source Code Representation

- [Prologue](./part-3-intro.md)
- [Command-line arguments](./cli.md)
- [The Rustc Driver and Interface](./rustc-driver.md)
    - [Ex: Type checking through `rustc_interface`](./rustc-driver-interacting-with-the-ast.md)
    - [Ex: Getting diagnostics through `rustc_interface`](./rustc-driver-getting-diagnostics.md)
- [Syntax and the AST](./syntax-intro.md)
    - [Lexing and Parsing](./the-parser.md)
    - [Macro expansion](./macro-expansion.md)
    - [Name resolution](./name-resolution.md)
    - [`#[test]` Implementation](./test-implementation.md)
    - [Panic Implementation](./panic-implementation.md)
    - [AST Validation](./ast-validation.md)
    - [Feature Gate Checking](./feature-gate-ck.md)
- [The HIR (High-level IR)](./hir.md)
    - [Lowering AST to HIR](./lowering.md)
    - [Debugging](./hir-debugging.md)
- [The MIR (Mid-level IR)](./mir/index.md)
    - [THIR and MIR construction](./mir/construction.md)
    - [MIR visitor and traversal](./mir/visitor.md)
    - [MIR passes: getting the MIR for a function](./mir/passes.md)
- [Identifiers in the Compiler](./identifiers.md)
- [Closure expansion](./closure.md)

# Analysis

- [Prologue](./part-4-intro.md)
- [The `ty` module: representing types](./ty.md)
    - [Generics and substitutions](./generics.md)
    - [`TypeFolder` and `TypeFoldable`](./ty-fold.md)
    - [Generic arguments](./generic_arguments.md)
- [Type inference](./type-inference.md)
- [Trait solving](./traits/resolution.md)
    - [Early and Late Bound Parameters](./early-late-bound.md)
    - [Higher-ranked trait bounds](./traits/hrtb.md)
    - [Caching subtleties](./traits/caching.md)
    - [Specialization](./traits/specialization.md)
    - [Chalk-based trait solving](./traits/chalk.md)
        - [Lowering to logic](./traits/lowering-to-logic.md)
        - [Goals and clauses](./traits/goals-and-clauses.md)
        - [Canonical queries](./traits/canonical-queries.md)
- [Type checking](./type-checking.md)
    - [Method Lookup](./method-lookup.md)
    - [Variance](./variance.md)
    - [Opaque Types](./opaque-types-type-alias-impl-trait.md)
- [Pattern and Exhaustiveness Checking](./pat-exhaustive-checking.md)
- [MIR dataflow](./mir/dataflow.md)
- [The borrow checker](./borrow_check.md)
    - [Tracking moves and initialization](./borrow_check/moves_and_initialization.md)
        - [Move paths](./borrow_check/moves_and_initialization/move_paths.md)
    - [MIR type checker](./borrow_check/type_check.md)
    - [Region inference](./borrow_check/region_inference.md)
        - [Constraint propagation](./borrow_check/region_inference/constraint_propagation.md)
        - [Lifetime parameters](./borrow_check/region_inference/lifetime_parameters.md)
        - [Member constraints](./borrow_check/region_inference/member_constraints.md)
        - [Placeholders and universes][pau]
        - [Closure constraints](./borrow_check/region_inference/closure_constraints.md)
        - [Error reporting](./borrow_check/region_inference/error_reporting.md)
    - [Two-phase-borrows](./borrow_check/two_phase_borrows.md)
- [Parameter Environments](./param_env.md)
- [Errors and Lints](diagnostics.md)
    - [Creating Errors With SessionDiagnostic](./diagnostics/sessiondiagnostic.md)
    - [`LintStore`](./diagnostics/lintstore.md)
    - [Diagnostic Codes](./diagnostics/diagnostic-codes.md)

# MIR to Binaries

- [Prologue](./part-5-intro.md)
- [MIR optimizations](./mir/optimizations.md)
- [Debugging](./mir/debugging.md)
- [Constant evaluation](./const-eval.md)
    - [miri const evaluator](./miri.md)
- [Monomorphization](./backend/monomorph.md)
- [Lowering MIR](./backend/lowering-mir.md)
- [Code Generation](./backend/codegen.md)
    - [Updating LLVM](./backend/updating-llvm.md)
    - [Debugging LLVM](./backend/debugging.md)
    - [Backend Agnostic Codegen](./backend/backend-agnostic.md)
    - [Implicit Caller Location](./backend/implicit-caller-location.md)
- [Libraries and Metadata](./backend/libs-and-metadata.md)
- [Profile-guided Optimization](./profile-guided-optimization.md)
- [LLVM Source-Based Code Coverage](./llvm-coverage-instrumentation.md)
- [Sanitizers Support](./sanitizers.md)
- [Debugging Support in the Rust Compiler](./debugging-support-in-rustc.md)

---

[Appendix A: Background topics](./appendix/background.md)
[Appendix B: Glossary](./appendix/glossary.md)
[Appendix C: Code Index](./appendix/code-index.md)
[Appendix D: Compiler Lecture Series](./appendix/compiler-lecture.md)
[Appendix E: Bibliography](./appendix/bibliography.md)

[Appendix Z: HumorRust](./appendix/humorust.md)

---

[pau]: ./borrow_check/region_inference/placeholders_and_universes.md
