//! Generated file, do not edit by hand, see `xtask/src/codegen`

pub struct Lint {
    pub label: &'static str,
    pub description: &'static str,
}

pub const DEFAULT_LINTS: &[Lint] = &[
    Lint {
        label: "absolute_paths_not_starting_with_crate",
        description: r##"fully qualified paths that start with a module name instead of `crate`, `self`, or an extern crate name"##,
    },
    Lint { label: "ambiguous_associated_items", description: r##"ambiguous associated items"## },
    Lint { label: "anonymous_parameters", description: r##"detects anonymous parameters"## },
    Lint { label: "arithmetic_overflow", description: r##"arithmetic operation overflows"## },
    Lint { label: "array_into_iter", description: r##"detects calling `into_iter` on arrays"## },
    Lint {
        label: "asm_sub_register",
        description: r##"using only a subset of a register for inline asm inputs"##,
    },
    Lint { label: "bad_asm_style", description: r##"incorrect use of inline assembly"## },
    Lint {
        label: "bare_trait_objects",
        description: r##"suggest using `dyn Trait` for trait objects"##,
    },
    Lint {
        label: "bindings_with_variant_name",
        description: r##"detects pattern bindings with the same name as one of the matched variants"##,
    },
    Lint { label: "box_pointers", description: r##"use of owned (Box type) heap memory"## },
    Lint {
        label: "cenum_impl_drop_cast",
        description: r##"a C-like enum implementing Drop is cast"##,
    },
    Lint {
        label: "clashing_extern_declarations",
        description: r##"detects when an extern fn has been declared with the same name but different types"##,
    },
    Lint {
        label: "coherence_leak_check",
        description: r##"distinct impls distinguished only by the leak-check code"##,
    },
    Lint {
        label: "conflicting_repr_hints",
        description: r##"conflicts between `#[repr(..)]` hints that were previously accepted and used in practice"##,
    },
    Lint {
        label: "confusable_idents",
        description: r##"detects visually confusable pairs between identifiers"##,
    },
    Lint {
        label: "const_err",
        description: r##"constant evaluation encountered erroneous expression"##,
    },
    Lint {
        label: "const_evaluatable_unchecked",
        description: r##"detects a generic constant is used in a type without a emitting a warning"##,
    },
    Lint {
        label: "const_item_mutation",
        description: r##"detects attempts to mutate a `const` item"##,
    },
    Lint { label: "dead_code", description: r##"detect unused, unexported items"## },
    Lint { label: "deprecated", description: r##"detects use of deprecated items"## },
    Lint {
        label: "deprecated_in_future",
        description: r##"detects use of items that will be deprecated in a future version"##,
    },
    Lint {
        label: "deref_nullptr",
        description: r##"detects when an null pointer is dereferenced"##,
    },
    Lint {
        label: "disjoint_capture_migration",
        description: r##"Drop reorder and auto traits error because of `capture_disjoint_fields`"##,
    },
    Lint { label: "drop_bounds", description: r##"bounds of the form `T: Drop` are useless"## },
    Lint {
        label: "elided_lifetimes_in_paths",
        description: r##"hidden lifetime parameters in types are deprecated"##,
    },
    Lint {
        label: "ellipsis_inclusive_range_patterns",
        description: r##"`...` range patterns are deprecated"##,
    },
    Lint {
        label: "explicit_outlives_requirements",
        description: r##"outlives requirements can be inferred"##,
    },
    Lint {
        label: "exported_private_dependencies",
        description: r##"public interface leaks type from a private dependency"##,
    },
    Lint { label: "forbidden_lint_groups", description: r##"applying forbid to lint-groups"## },
    Lint {
        label: "function_item_references",
        description: r##"suggest casting to a function pointer when attempting to take references to function items"##,
    },
    Lint {
        label: "future_incompatible",
        description: r##"lint group for: keyword-idents, anonymous-parameters, ellipsis-inclusive-range-patterns, forbidden-lint-groups, illegal-floating-point-literal-pattern, private-in-public, pub-use-of-private-extern-crate, invalid-type-param-default, const-err, unaligned-references, patterns-in-fns-without-body, missing-fragment-specifier, late-bound-lifetime-arguments, order-dependent-trait-objects, coherence-leak-check, tyvar-behind-raw-pointer, bare-trait-objects, absolute-paths-not-starting-with-crate, unstable-name-collisions, where-clauses-object-safety, proc-macro-derive-resolution-fallback, macro-expanded-macro-exports-accessed-by-absolute-paths, ill-formed-attribute-input, conflicting-repr-hints, ambiguous-associated-items, mutable-borrow-reservation-conflict, indirect-structural-match, pointer-structural-match, nontrivial-structural-match, soft-unstable, cenum-impl-drop-cast, const-evaluatable-unchecked, uninhabited-static, unsupported-naked-functions, semicolon-in-expressions-from-macros, legacy-derive-helpers, proc-macro-back-compat, array-into-iter"##,
    },
    Lint {
        label: "ill_formed_attribute_input",
        description: r##"ill-formed attribute inputs that were previously accepted and used in practice"##,
    },
    Lint {
        label: "illegal_floating_point_literal_pattern",
        description: r##"floating-point literals cannot be used in patterns"##,
    },
    Lint {
        label: "improper_ctypes",
        description: r##"proper use of libc types in foreign modules"##,
    },
    Lint {
        label: "improper_ctypes_definitions",
        description: r##"proper use of libc types in foreign item definitions"##,
    },
    Lint {
        label: "incomplete_features",
        description: r##"incomplete features that may function improperly in some or all cases"##,
    },
    Lint { label: "incomplete_include", description: r##"trailing content in included file"## },
    Lint {
        label: "indirect_structural_match",
        description: r##"constant used in pattern contains value of non-structural-match type in a field or a variant"##,
    },
    Lint {
        label: "ineffective_unstable_trait_impl",
        description: r##"detects `#[unstable]` on stable trait implementations for stable types"##,
    },
    Lint {
        label: "inline_no_sanitize",
        description: r##"detects incompatible use of `#[inline(always)]` and `#[no_sanitize(...)]`"##,
    },
    Lint {
        label: "invalid_type_param_default",
        description: r##"type parameter default erroneously allowed in invalid location"##,
    },
    Lint {
        label: "invalid_value",
        description: r##"an invalid value is being created (such as a null reference)"##,
    },
    Lint {
        label: "irrefutable_let_patterns",
        description: r##"detects irrefutable patterns in `if let` and `while let` statements"##,
    },
    Lint {
        label: "keyword_idents",
        description: r##"detects edition keywords being used as an identifier"##,
    },
    Lint { label: "large_assignments", description: r##"detects large moves or copies"## },
    Lint {
        label: "late_bound_lifetime_arguments",
        description: r##"detects generic lifetime arguments in path segments with late bound lifetime parameters"##,
    },
    Lint {
        label: "legacy_derive_helpers",
        description: r##"detects derive helper attributes that are used before they are introduced"##,
    },
    Lint {
        label: "macro_expanded_macro_exports_accessed_by_absolute_paths",
        description: r##"macro-expanded `macro_export` macros from the current crate cannot be referred to by absolute paths"##,
    },
    Lint {
        label: "macro_use_extern_crate",
        description: r##"the `#[macro_use]` attribute is now deprecated in favor of using macros via the module system"##,
    },
    Lint {
        label: "meta_variable_misuse",
        description: r##"possible meta-variable misuse at macro definition"##,
    },
    Lint { label: "missing_abi", description: r##"No declared ABI for extern declaration"## },
    Lint {
        label: "missing_copy_implementations",
        description: r##"detects potentially-forgotten implementations of `Copy`"##,
    },
    Lint {
        label: "missing_debug_implementations",
        description: r##"detects missing implementations of Debug"##,
    },
    Lint {
        label: "missing_docs",
        description: r##"detects missing documentation for public members"##,
    },
    Lint {
        label: "missing_fragment_specifier",
        description: r##"detects missing fragment specifiers in unused `macro_rules!` patterns"##,
    },
    Lint {
        label: "mixed_script_confusables",
        description: r##"detects Unicode scripts whose mixed script confusables codepoints are solely used"##,
    },
    Lint {
        label: "mutable_borrow_reservation_conflict",
        description: r##"reservation of a two-phased borrow conflicts with other shared borrows"##,
    },
    Lint {
        label: "mutable_transmutes",
        description: r##"mutating transmuted &mut T from &T may cause undefined behavior"##,
    },
    Lint {
        label: "no_mangle_const_items",
        description: r##"const items will not have their symbols exported"##,
    },
    Lint { label: "no_mangle_generic_items", description: r##"generic items must be mangled"## },
    Lint { label: "non_ascii_idents", description: r##"detects non-ASCII identifiers"## },
    Lint {
        label: "non_camel_case_types",
        description: r##"types, variants, traits and type parameters should have camel case names"##,
    },
    Lint {
        label: "non_fmt_panic",
        description: r##"detect single-argument panic!() invocations in which the argument is not a format string"##,
    },
    Lint {
        label: "non_shorthand_field_patterns",
        description: r##"using `Struct { x: x }` instead of `Struct { x }` in a pattern"##,
    },
    Lint {
        label: "non_snake_case",
        description: r##"variables, methods, functions, lifetime parameters and modules should have snake case names"##,
    },
    Lint {
        label: "non_upper_case_globals",
        description: r##"static constants should have uppercase identifiers"##,
    },
    Lint {
        label: "nonstandard_style",
        description: r##"lint group for: non-camel-case-types, non-snake-case, non-upper-case-globals"##,
    },
    Lint {
        label: "nontrivial_structural_match",
        description: r##"constant used in pattern of non-structural-match type and the constant's initializer expression contains values of non-structural-match types"##,
    },
    Lint {
        label: "noop_method_call",
        description: r##"detects the use of well-known noop methods"##,
    },
    Lint {
        label: "or_patterns_back_compat",
        description: r##"detects usage of old versions of or-patterns"##,
    },
    Lint {
        label: "order_dependent_trait_objects",
        description: r##"trait-object types were treated as different depending on marker-trait order"##,
    },
    Lint { label: "overflowing_literals", description: r##"literal out of range for its type"## },
    Lint {
        label: "overlapping_range_endpoints",
        description: r##"detects range patterns with overlapping endpoints"##,
    },
    Lint { label: "path_statements", description: r##"path statements with no effect"## },
    Lint {
        label: "patterns_in_fns_without_body",
        description: r##"patterns in functions without body were erroneously allowed"##,
    },
    Lint {
        label: "pointer_structural_match",
        description: r##"pointers are not structural-match"##,
    },
    Lint {
        label: "private_in_public",
        description: r##"detect private items in public interfaces not caught by the old implementation"##,
    },
    Lint {
        label: "proc_macro_back_compat",
        description: r##"detects usage of old versions of certain proc-macro crates"##,
    },
    Lint {
        label: "proc_macro_derive_resolution_fallback",
        description: r##"detects proc macro derives using inaccessible names from parent modules"##,
    },
    Lint {
        label: "pub_use_of_private_extern_crate",
        description: r##"detect public re-exports of private extern crates"##,
    },
    Lint {
        label: "redundant_semicolons",
        description: r##"detects unnecessary trailing semicolons"##,
    },
    Lint {
        label: "renamed_and_removed_lints",
        description: r##"lints that have been renamed or removed"##,
    },
    Lint {
        label: "rust_2018_compatibility",
        description: r##"lint group for: keyword-idents, anonymous-parameters, tyvar-behind-raw-pointer, absolute-paths-not-starting-with-crate"##,
    },
    Lint {
        label: "rust_2018_idioms",
        description: r##"lint group for: bare-trait-objects, unused-extern-crates, ellipsis-inclusive-range-patterns, elided-lifetimes-in-paths, explicit-outlives-requirements"##,
    },
    Lint {
        label: "rust_2021_compatibility",
        description: r##"lint group for: ellipsis-inclusive-range-patterns, bare-trait-objects"##,
    },
    Lint {
        label: "semicolon_in_expressions_from_macros",
        description: r##"trailing semicolon in macro body used as expression"##,
    },
    Lint {
        label: "single_use_lifetimes",
        description: r##"detects lifetime parameters that are only used once"##,
    },
    Lint {
        label: "soft_unstable",
        description: r##"a feature gate that doesn't break dependent crates"##,
    },
    Lint {
        label: "stable_features",
        description: r##"stable features found in `#[feature]` directive"##,
    },
    Lint {
        label: "temporary_cstring_as_ptr",
        description: r##"detects getting the inner pointer of a temporary `CString`"##,
    },
    Lint {
        label: "trivial_bounds",
        description: r##"these bounds don't depend on an type parameters"##,
    },
    Lint {
        label: "trivial_casts",
        description: r##"detects trivial casts which could be removed"##,
    },
    Lint {
        label: "trivial_numeric_casts",
        description: r##"detects trivial casts of numeric types which could be removed"##,
    },
    Lint {
        label: "type_alias_bounds",
        description: r##"bounds in type aliases are not enforced"##,
    },
    Lint {
        label: "tyvar_behind_raw_pointer",
        description: r##"raw pointer to an inference variable"##,
    },
    Lint {
        label: "unaligned_references",
        description: r##"detects unaligned references to fields of packed structs"##,
    },
    Lint {
        label: "uncommon_codepoints",
        description: r##"detects uncommon Unicode codepoints in identifiers"##,
    },
    Lint {
        label: "unconditional_panic",
        description: r##"operation will cause a panic at runtime"##,
    },
    Lint {
        label: "unconditional_recursion",
        description: r##"functions that cannot return without calling themselves"##,
    },
    Lint { label: "uninhabited_static", description: r##"uninhabited static"## },
    Lint {
        label: "unknown_crate_types",
        description: r##"unknown crate type found in `#[crate_type]` directive"##,
    },
    Lint { label: "unknown_lints", description: r##"unrecognized lint attribute"## },
    Lint {
        label: "unnameable_test_items",
        description: r##"detects an item that cannot be named being marked as `#[test_case]`"##,
    },
    Lint { label: "unreachable_code", description: r##"detects unreachable code paths"## },
    Lint { label: "unreachable_patterns", description: r##"detects unreachable patterns"## },
    Lint {
        label: "unreachable_pub",
        description: r##"`pub` items not reachable from crate root"##,
    },
    Lint { label: "unsafe_code", description: r##"usage of `unsafe` code"## },
    Lint {
        label: "unsafe_op_in_unsafe_fn",
        description: r##"unsafe operations in unsafe functions without an explicit unsafe block are deprecated"##,
    },
    Lint {
        label: "unstable_features",
        description: r##"enabling unstable features (deprecated. do not use)"##,
    },
    Lint {
        label: "unstable_name_collisions",
        description: r##"detects name collision with an existing but unstable method"##,
    },
    Lint {
        label: "unsupported_naked_functions",
        description: r##"unsupported naked function definitions"##,
    },
    Lint {
        label: "unused",
        description: r##"lint group for: unused-imports, unused-variables, unused-assignments, dead-code, unused-mut, unreachable-code, unreachable-patterns, unused-must-use, unused-unsafe, path-statements, unused-attributes, unused-macros, unused-allocation, unused-doc-comments, unused-extern-crates, unused-features, unused-labels, unused-parens, unused-braces, redundant-semicolons"##,
    },
    Lint {
        label: "unused_allocation",
        description: r##"detects unnecessary allocations that can be eliminated"##,
    },
    Lint {
        label: "unused_assignments",
        description: r##"detect assignments that will never be read"##,
    },
    Lint {
        label: "unused_attributes",
        description: r##"detects attributes that were not used by the compiler"##,
    },
    Lint { label: "unused_braces", description: r##"unnecessary braces around an expression"## },
    Lint {
        label: "unused_comparisons",
        description: r##"comparisons made useless by limits of the types involved"##,
    },
    Lint {
        label: "unused_crate_dependencies",
        description: r##"crate dependencies that are never used"##,
    },
    Lint {
        label: "unused_doc_comments",
        description: r##"detects doc comments that aren't used by rustdoc"##,
    },
    Lint { label: "unused_extern_crates", description: r##"extern crates that are never used"## },
    Lint {
        label: "unused_features",
        description: r##"unused features found in crate-level `#[feature]` directives"##,
    },
    Lint {
        label: "unused_import_braces",
        description: r##"unnecessary braces around an imported item"##,
    },
    Lint { label: "unused_imports", description: r##"imports that are never used"## },
    Lint { label: "unused_labels", description: r##"detects labels that are never used"## },
    Lint {
        label: "unused_lifetimes",
        description: r##"detects lifetime parameters that are never used"##,
    },
    Lint { label: "unused_macros", description: r##"detects macros that were not used"## },
    Lint {
        label: "unused_must_use",
        description: r##"unused result of a type flagged as `#[must_use]`"##,
    },
    Lint {
        label: "unused_mut",
        description: r##"detect mut variables which don't need to be mutable"##,
    },
    Lint {
        label: "unused_parens",
        description: r##"`if`, `match`, `while` and `return` do not need parentheses"##,
    },
    Lint {
        label: "unused_qualifications",
        description: r##"detects unnecessarily qualified names"##,
    },
    Lint {
        label: "unused_results",
        description: r##"unused result of an expression in a statement"##,
    },
    Lint { label: "unused_unsafe", description: r##"unnecessary use of an `unsafe` block"## },
    Lint {
        label: "unused_variables",
        description: r##"detect variables which are not used in any way"##,
    },
    Lint {
        label: "useless_deprecated",
        description: r##"detects deprecation attributes with no effect"##,
    },
    Lint {
        label: "variant_size_differences",
        description: r##"detects enums with widely varying variant sizes"##,
    },
    Lint {
        label: "warnings",
        description: r##"mass-change the level for lints which produce warnings"##,
    },
    Lint {
        label: "warnings",
        description: r##"lint group for: all lints that are set to issue warnings"##,
    },
    Lint {
        label: "where_clauses_object_safety",
        description: r##"checks the object safety of where clauses"##,
    },
    Lint {
        label: "while_true",
        description: r##"suggest using `loop { }` instead of `while true { }`"##,
    },
];

pub const FEATURES: &[Lint] = &[
    Lint {
        label: "abi_c_cmse_nonsecure_call",
        description: r##"# `abi_c_cmse_nonsecure_call`

The tracking issue for this feature is: [#81391]

[#81391]: https://github.com/rust-lang/rust/issues/81391

------------------------

The [TrustZone-M
feature](https://developer.arm.com/documentation/100690/latest/) is available
for targets with the Armv8-M architecture profile (`thumbv8m` in their target
name).
LLVM, the Rust compiler and the linker are providing
[support](https://developer.arm.com/documentation/ecm0359818/latest/) for the
TrustZone-M feature.

One of the things provided, with this unstable feature, is the
`C-cmse-nonsecure-call` function ABI. This ABI is used on function pointers to
non-secure code to mark a non-secure function call (see [section
5.5](https://developer.arm.com/documentation/ecm0359818/latest/) for details).

With this ABI, the compiler will do the following to perform the call:
* save registers needed after the call to Secure memory
* clear all registers that might contain confidential information
* clear the Least Significant Bit of the function address
* branches using the BLXNS instruction

To avoid using the non-secure stack, the compiler will constrain the number and
type of parameters/return value.

The `extern "C-cmse-nonsecure-call"` ABI is otherwise equivalent to the
`extern "C"` ABI.

<!-- NOTE(ignore) this example is specific to thumbv8m targets -->

``` rust,ignore
#![no_std]
#![feature(abi_c_cmse_nonsecure_call)]

#[no_mangle]
pub fn call_nonsecure_function(addr: usize) -> u32 {
    let non_secure_function =
        unsafe { core::mem::transmute::<usize, extern "C-cmse-nonsecure-call" fn() -> u32>(addr) };
    non_secure_function()
}
```

``` text
$ rustc --emit asm --crate-type lib --target thumbv8m.main-none-eabi function.rs

call_nonsecure_function:
        .fnstart
        .save   {r7, lr}
        push    {r7, lr}
        .setfp  r7, sp
        mov     r7, sp
        .pad    #16
        sub     sp, #16
        str     r0, [sp, #12]
        ldr     r0, [sp, #12]
        str     r0, [sp, #8]
        b       .LBB0_1
.LBB0_1:
        ldr     r0, [sp, #8]
        push.w  {r4, r5, r6, r7, r8, r9, r10, r11}
        bic     r0, r0, #1
        mov     r1, r0
        mov     r2, r0
        mov     r3, r0
        mov     r4, r0
        mov     r5, r0
        mov     r6, r0
        mov     r7, r0
        mov     r8, r0
        mov     r9, r0
        mov     r10, r0
        mov     r11, r0
        mov     r12, r0
        msr     apsr_nzcvq, r0
        blxns   r0
        pop.w   {r4, r5, r6, r7, r8, r9, r10, r11}
        str     r0, [sp, #4]
        b       .LBB0_2
.LBB0_2:
        ldr     r0, [sp, #4]
        add     sp, #16
        pop     {r7, pc}
```
"##,
    },
    Lint {
        label: "abi_msp430_interrupt",
        description: r##"# `abi_msp430_interrupt`

The tracking issue for this feature is: [#38487]

[#38487]: https://github.com/rust-lang/rust/issues/38487

------------------------

In the MSP430 architecture, interrupt handlers have a special calling
convention. You can use the `"msp430-interrupt"` ABI to make the compiler apply
the right calling convention to the interrupt handlers you define.

<!-- NOTE(ignore) this example is specific to the msp430 target -->

``` rust,ignore
#![feature(abi_msp430_interrupt)]
#![no_std]

// Place the interrupt handler at the appropriate memory address
// (Alternatively, you can use `#[used]` and remove `pub` and `#[no_mangle]`)
#[link_section = "__interrupt_vector_10"]
#[no_mangle]
pub static TIM0_VECTOR: extern "msp430-interrupt" fn() = tim0;

// The interrupt handler
extern "msp430-interrupt" fn tim0() {
    // ..
}
```

``` text
$ msp430-elf-objdump -CD ./target/msp430/release/app
Disassembly of section __interrupt_vector_10:

0000fff2 <TIM0_VECTOR>:
    fff2:       00 c0           interrupt service routine at 0xc000

Disassembly of section .text:

0000c000 <int::tim0>:
    c000:       00 13           reti
```
"##,
    },
    Lint {
        label: "abi_ptx",
        description: r##"# `abi_ptx`

The tracking issue for this feature is: [#38788]

[#38788]: https://github.com/rust-lang/rust/issues/38788

------------------------

When emitting PTX code, all vanilla Rust functions (`fn`) get translated to
"device" functions. These functions are *not* callable from the host via the
CUDA API so a crate with only device functions is not too useful!

OTOH, "global" functions *can* be called by the host; you can think of them
as the real public API of your crate. To produce a global function use the
`"ptx-kernel"` ABI.

<!-- NOTE(ignore) this example is specific to the nvptx targets -->

``` rust,ignore
#![feature(abi_ptx)]
#![no_std]

pub unsafe extern "ptx-kernel" fn global_function() {
    device_function();
}

pub fn device_function() {
    // ..
}
```

``` text
$ xargo rustc --target nvptx64-nvidia-cuda --release -- --emit=asm

$ cat $(find -name '*.s')
//
// Generated by LLVM NVPTX Back-End
//

.version 3.2
.target sm_20
.address_size 64

        // .globl       _ZN6kernel15global_function17h46111ebe6516b382E

.visible .entry _ZN6kernel15global_function17h46111ebe6516b382E()
{


        ret;
}

        // .globl       _ZN6kernel15device_function17hd6a0e4993bbf3f78E
.visible .func _ZN6kernel15device_function17hd6a0e4993bbf3f78E()
{


        ret;
}
```
"##,
    },
    Lint {
        label: "abi_thiscall",
        description: r##"# `abi_thiscall`

The tracking issue for this feature is: [#42202]

[#42202]: https://github.com/rust-lang/rust/issues/42202

------------------------

The MSVC ABI on x86 Windows uses the `thiscall` calling convention for C++
instance methods by default; it is identical to the usual (C) calling
convention on x86 Windows except that the first parameter of the method,
the `this` pointer, is passed in the ECX register.
"##,
    },
    Lint {
        label: "allocator_api",
        description: r##"# `allocator_api`

The tracking issue for this feature is [#32838]

[#32838]: https://github.com/rust-lang/rust/issues/32838

------------------------

Sometimes you want the memory for one collection to use a different
allocator than the memory for another collection. In this case,
replacing the global allocator is not a workable option. Instead,
you need to pass in an instance of an `AllocRef` to each collection
for which you want a custom allocator.

TBD
"##,
    },
    Lint {
        label: "allocator_internals",
        description: r##"# `allocator_internals`

This feature does not have a tracking issue, it is an unstable implementation
detail of the `global_allocator` feature not intended for use outside the
compiler.

------------------------
"##,
    },
    Lint {
        label: "arbitrary_enum_discriminant",
        description: r##"# `arbitrary_enum_discriminant`

The tracking issue for this feature is: [#60553]

[#60553]: https://github.com/rust-lang/rust/issues/60553

------------------------

The `arbitrary_enum_discriminant` feature permits tuple-like and
struct-like enum variants with `#[repr(<int-type>)]` to have explicit discriminants.

## Examples

```rust
#![feature(arbitrary_enum_discriminant)]

#[allow(dead_code)]
#[repr(u8)]
enum Enum {
    Unit = 3,
    Tuple(u16) = 2,
    Struct {
        a: u8,
        b: u16,
    } = 1,
}

impl Enum {
    fn tag(&self) -> u8 {
        unsafe { *(self as *const Self as *const u8) }
    }
}

assert_eq!(3, Enum::Unit.tag());
assert_eq!(2, Enum::Tuple(5).tag());
assert_eq!(1, Enum::Struct{a: 7, b: 11}.tag());
```
"##,
    },
    Lint {
        label: "asm",
        description: r##"# `asm`

The tracking issue for this feature is: [#72016]

[#72016]: https://github.com/rust-lang/rust/issues/72016

------------------------

For extremely low-level manipulations and performance reasons, one
might wish to control the CPU directly. Rust supports using inline
assembly to do this via the `asm!` macro.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Rust provides support for inline assembly via the `asm!` macro.
It can be used to embed handwritten assembly in the assembly output generated by the compiler.
Generally this should not be necessary, but might be where the required performance or timing
cannot be otherwise achieved. Accessing low level hardware primitives, e.g. in kernel code, may also demand this functionality.

> **Note**: the examples here are given in x86/x86-64 assembly, but other architectures are also supported.

Inline assembly is currently supported on the following architectures:
- x86 and x86-64
- ARM
- AArch64
- RISC-V
- NVPTX
- PowerPC
- Hexagon
- MIPS32r2 and MIPS64r2
- wasm32

## Basic usage

Let us start with the simplest possible example:

```rust,allow_fail
#![feature(asm)]
unsafe {
    asm!("nop");
}
```

This will insert a NOP (no operation) instruction into the assembly generated by the compiler.
Note that all `asm!` invocations have to be inside an `unsafe` block, as they could insert
arbitrary instructions and break various invariants. The instructions to be inserted are listed
in the first argument of the `asm!` macro as a string literal.

## Inputs and outputs

Now inserting an instruction that does nothing is rather boring. Let us do something that
actually acts on data:

```rust,allow_fail
#![feature(asm)]
let x: u64;
unsafe {
    asm!("mov {}, 5", out(reg) x);
}
assert_eq!(x, 5);
```

This will write the value `5` into the `u64` variable `x`.
You can see that the string literal we use to specify instructions is actually a template string.
It is governed by the same rules as Rust [format strings][format-syntax].
The arguments that are inserted into the template however look a bit different then you may
be familiar with. First we need to specify if the variable is an input or an output of the
inline assembly. In this case it is an output. We declared this by writing `out`.
We also need to specify in what kind of register the assembly expects the variable.
In this case we put it in an arbitrary general purpose register by specifying `reg`.
The compiler will choose an appropriate register to insert into
the template and will read the variable from there after the inline assembly finishes executing.

Let us see another example that also uses an input:

```rust,allow_fail
#![feature(asm)]
let i: u64 = 3;
let o: u64;
unsafe {
    asm!(
        "mov {0}, {1}",
        "add {0}, {number}",
        out(reg) o,
        in(reg) i,
        number = const 5,
    );
}
assert_eq!(o, 8);
```

This will add `5` to the input in variable `i` and write the result to variable `o`.
The particular way this assembly does this is first copying the value from `i` to the output,
and then adding `5` to it.

The example shows a few things:

First, we can see that `asm!` allows multiple template string arguments; each
one is treated as a separate line of assembly code, as if they were all joined
together with newlines between them. This makes it easy to format assembly
code.

Second, we can see that inputs are declared by writing `in` instead of `out`.

Third, one of our operands has a type we haven't seen yet, `const`.
This tells the compiler to expand this argument to value directly inside the assembly template.
This is only possible for constants and literals.

Fourth, we can see that we can specify an argument number, or name as in any format string.
For inline assembly templates this is particularly useful as arguments are often used more than once.
For more complex inline assembly using this facility is generally recommended, as it improves
readability, and allows reordering instructions without changing the argument order.

We can further refine the above example to avoid the `mov` instruction:

```rust,allow_fail
#![feature(asm)]
let mut x: u64 = 3;
unsafe {
    asm!("add {0}, {number}", inout(reg) x, number = const 5);
}
assert_eq!(x, 8);
```

We can see that `inout` is used to specify an argument that is both input and output.
This is different from specifying an input and output separately in that it is guaranteed to assign both to the same register.

It is also possible to specify different variables for the input and output parts of an `inout` operand:

```rust,allow_fail
#![feature(asm)]
let x: u64 = 3;
let y: u64;
unsafe {
    asm!("add {0}, {number}", inout(reg) x => y, number = const 5);
}
assert_eq!(y, 8);
```

## Late output operands

The Rust compiler is conservative with its allocation of operands. It is assumed that an `out`
can be written at any time, and can therefore not share its location with any other argument.
However, to guarantee optimal performance it is important to use as few registers as possible,
so they won't have to be saved and reloaded around the inline assembly block.
To achieve this Rust provides a `lateout` specifier. This can be used on any output that is
written only after all inputs have been consumed.
There is also a `inlateout` variant of this specifier.

Here is an example where `inlateout` *cannot* be used:

```rust,allow_fail
#![feature(asm)]
let mut a: u64 = 4;
let b: u64 = 4;
let c: u64 = 4;
unsafe {
    asm!(
        "add {0}, {1}",
        "add {0}, {2}",
        inout(reg) a,
        in(reg) b,
        in(reg) c,
    );
}
assert_eq!(a, 12);
```

Here the compiler is free to allocate the same register for inputs `b` and `c` since it knows they have the same value. However it must allocate a separate register for `a` since it uses `inout` and not `inlateout`. If `inlateout` was used, then `a` and `c` could be allocated to the same register, in which case the first instruction to overwrite the value of `c` and cause the assembly code to produce the wrong result.

However the following example can use `inlateout` since the output is only modified after all input registers have been read:

```rust,allow_fail
#![feature(asm)]
let mut a: u64 = 4;
let b: u64 = 4;
unsafe {
    asm!("add {0}, {1}", inlateout(reg) a, in(reg) b);
}
assert_eq!(a, 8);
```

As you can see, this assembly fragment will still work correctly if `a` and `b` are assigned to the same register.

## Explicit register operands

Some instructions require that the operands be in a specific register.
Therefore, Rust inline assembly provides some more specific constraint specifiers.
While `reg` is generally available on any architecture, these are highly architecture specific. E.g. for x86 the general purpose registers `eax`, `ebx`, `ecx`, `edx`, `ebp`, `esi`, and `edi`
among others can be addressed by their name.

```rust,allow_fail,no_run
#![feature(asm)]
let cmd = 0xd1;
unsafe {
    asm!("out 0x64, eax", in("eax") cmd);
}
```

In this example we call the `out` instruction to output the content of the `cmd` variable
to port `0x64`. Since the `out` instruction only accepts `eax` (and its sub registers) as operand
we had to use the `eax` constraint specifier.

Note that unlike other operand types, explicit register operands cannot be used in the template string: you can't use `{}` and should write the register name directly instead. Also, they must appear at the end of the operand list after all other operand types.

Consider this example which uses the x86 `mul` instruction:

```rust,allow_fail
#![feature(asm)]
fn mul(a: u64, b: u64) -> u128 {
    let lo: u64;
    let hi: u64;

    unsafe {
        asm!(
            // The x86 mul instruction takes rax as an implicit input and writes
            // the 128-bit result of the multiplication to rax:rdx.
            "mul {}",
            in(reg) a,
            inlateout("rax") b => lo,
            lateout("rdx") hi
        );
    }

    ((hi as u128) << 64) + lo as u128
}
```

This uses the `mul` instruction to multiply two 64-bit inputs with a 128-bit result.
The only explicit operand is a register, that we fill from the variable `a`.
The second operand is implicit, and must be the `rax` register, which we fill from the variable `b`.
The lower 64 bits of the result are stored in `rax` from which we fill the variable `lo`.
The higher 64 bits are stored in `rdx` from which we fill the variable `hi`.

## Clobbered registers

In many cases inline assembly will modify state that is not needed as an output.
Usually this is either because we have to use a scratch register in the assembly,
or instructions modify state that we don't need to further examine.
This state is generally referred to as being "clobbered".
We need to tell the compiler about this since it may need to save and restore this state
around the inline assembly block.

```rust,allow_fail
#![feature(asm)]
let ebx: u32;
let ecx: u32;

unsafe {
    asm!(
        "cpuid",
        // EAX 4 selects the "Deterministic Cache Parameters" CPUID leaf
        inout("eax") 4 => _,
        // ECX 0 selects the L0 cache information.
        inout("ecx") 0 => ecx,
        lateout("ebx") ebx,
        lateout("edx") _,
    );
}

println!(
    "L1 Cache: {}",
    ((ebx >> 22) + 1) * (((ebx >> 12) & 0x3ff) + 1) * ((ebx & 0xfff) + 1) * (ecx + 1)
);
```

In the example above we use the `cpuid` instruction to get the L1 cache size.
This instruction writes to `eax`, `ebx`, `ecx`, and `edx`, but for the cache size we only care about the contents of `ebx` and `ecx`.

However we still need to tell the compiler that `eax` and `edx` have been modified so that it can save any values that were in these registers before the asm. This is done by declaring these as outputs but with `_` instead of a variable name, which indicates that the output value is to be discarded.

This can also be used with a general register class (e.g. `reg`) to obtain a scratch register for use inside the asm code:

```rust,allow_fail
#![feature(asm)]
// Multiply x by 6 using shifts and adds
let mut x: u64 = 4;
unsafe {
    asm!(
        "mov {tmp}, {x}",
        "shl {tmp}, 1",
        "shl {x}, 2",
        "add {x}, {tmp}",
        x = inout(reg) x,
        tmp = out(reg) _,
    );
}
assert_eq!(x, 4 * 6);
```

## Symbol operands

A special operand type, `sym`, allows you to use the symbol name of a `fn` or `static` in inline assembly code.
This allows you to call a function or access a global variable without needing to keep its address in a register.

```rust,allow_fail
#![feature(asm)]
extern "C" fn foo(arg: i32) {
    println!("arg = {}", arg);
}

fn call_foo(arg: i32) {
    unsafe {
        asm!(
            "call {}",
            sym foo,
            // 1st argument in rdi, which is caller-saved
            inout("rdi") arg => _,
            // All caller-saved registers must be marked as clobbered
            out("rax") _, out("rcx") _, out("rdx") _, out("rsi") _,
            out("r8") _, out("r9") _, out("r10") _, out("r11") _,
            out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
            out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
            out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
            out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
            // Also mark AVX-512 registers as clobbered. This is accepted by the
            // compiler even if AVX-512 is not enabled on the current target.
            out("xmm16") _, out("xmm17") _, out("xmm18") _, out("xmm19") _,
            out("xmm20") _, out("xmm21") _, out("xmm22") _, out("xmm23") _,
            out("xmm24") _, out("xmm25") _, out("xmm26") _, out("xmm27") _,
            out("xmm28") _, out("xmm29") _, out("xmm30") _, out("xmm31") _,
        )
    }
}
```

Note that the `fn` or `static` item does not need to be public or `#[no_mangle]`:
the compiler will automatically insert the appropriate mangled symbol name into the assembly code.

## Register template modifiers

In some cases, fine control is needed over the way a register name is formatted when inserted into the template string. This is needed when an architecture's assembly language has several names for the same register, each typically being a "view" over a subset of the register (e.g. the low 32 bits of a 64-bit register).

By default the compiler will always choose the name that refers to the full register size (e.g. `rax` on x86-64, `eax` on x86, etc).

This default can be overriden by using modifiers on the template string operands, just like you would with format strings:

```rust,allow_fail
#![feature(asm)]
let mut x: u16 = 0xab;

unsafe {
    asm!("mov {0:h}, {0:l}", inout(reg_abcd) x);
}

assert_eq!(x, 0xabab);
```

In this example, we use the `reg_abcd` register class to restrict the register allocator to the 4 legacy x86 register (`ax`, `bx`, `cx`, `dx`) of which the first two bytes can be addressed independently.

Let us assume that the register allocator has chosen to allocate `x` in the `ax` register.
The `h` modifier will emit the register name for the high byte of that register and the `l` modifier will emit the register name for the low byte. The asm code will therefore be expanded as `mov ah, al` which copies the low byte of the value into the high byte.

If you use a smaller data type (e.g. `u16`) with an operand and forget the use template modifiers, the compiler will emit a warning and suggest the correct modifier to use.

## Memory address operands

Sometimes assembly instructions require operands passed via memory addresses/memory locations.
You have to manually use the memory address syntax specified by the respectively architectures.
For example, in x86/x86_64 and intel assembly syntax, you should wrap inputs/outputs in `[]`
to indicate they are memory operands:

```rust,allow_fail
#![feature(asm, llvm_asm)]
# fn load_fpu_control_word(control: u16) {
unsafe {
    asm!("fldcw [{}]", in(reg) &control, options(nostack));

    // Previously this would have been written with the deprecated `llvm_asm!` like this
    llvm_asm!("fldcw $0" :: "m" (control) :: "volatile");
}
# }
```

## Labels

The compiler is allowed to instantiate multiple copies an `asm!` block, for example when the function containing it is inlined in multiple places. As a consequence, you should only use GNU assembler [local labels] inside inline assembly code. Defining symbols in assembly code may lead to assembler and/or linker errors due to duplicate symbol definitions.

Moreover, due to [an llvm bug], you shouldn't use labels exclusively made of `0` and `1` digits, e.g. `0`, `11` or `101010`, as they may end up being interpreted as binary values.

```rust,allow_fail
#![feature(asm)]

let mut a = 0;
unsafe {
    asm!(
        "mov {0}, 10",
        "2:",
        "sub {0}, 1",
        "cmp {0}, 3",
        "jle 2f",
        "jmp 2b",
        "2:",
        "add {0}, 2",
        out(reg) a
    );
}
assert_eq!(a, 5);
```

This will decrement the `{0}` register value from 10 to 3, then add 2 and store it in `a`.

This example show a few thing:

First that the same number can be used as a label multiple times in the same inline block.

Second, that when a numeric label is used as a reference (as an instruction operand, for example), the suffixes b (“backward”) or f (“forward”) should be added to the numeric label. It will then refer to the nearest label defined by this number in this direction.

[local labels]: https://sourceware.org/binutils/docs/as/Symbol-Names.html#Local-Labels
[an llvm bug]: https://bugs.llvm.org/show_bug.cgi?id=36144

## Options

By default, an inline assembly block is treated the same way as an external FFI function call with a custom calling convention: it may read/write memory, have observable side effects, etc. However in many cases, it is desirable to give the compiler more information about what the assembly code is actually doing so that it can optimize better.

Let's take our previous example of an `add` instruction:

```rust,allow_fail
#![feature(asm)]
let mut a: u64 = 4;
let b: u64 = 4;
unsafe {
    asm!(
        "add {0}, {1}",
        inlateout(reg) a, in(reg) b,
        options(pure, nomem, nostack),
    );
}
assert_eq!(a, 8);
```

Options can be provided as an optional final argument to the `asm!` macro. We specified three options here:
- `pure` means that the asm code has no observable side effects and that its output depends only on its inputs. This allows the compiler optimizer to call the inline asm fewer times or even eliminate it entirely.
- `nomem` means that the asm code does not read or write to memory. By default the compiler will assume that inline assembly can read or write any memory address that is accessible to it (e.g. through a pointer passed as an operand, or a global).
- `nostack` means that the asm code does not push any data onto the stack. This allows the compiler to use optimizations such as the stack red zone on x86-64 to avoid stack pointer adjustments.

These allow the compiler to better optimize code using `asm!`, for example by eliminating pure `asm!` blocks whose outputs are not needed.

See the reference for the full list of available options and their effects.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Inline assembler is implemented as an unsafe macro `asm!()`.
The first argument to this macro is a template string literal used to build the final assembly.
The following arguments specify input and output operands.
When required, options are specified as the final argument.

The following ABNF specifies the general syntax:

```text
dir_spec := "in" / "out" / "lateout" / "inout" / "inlateout"
reg_spec := <register class> / "<explicit register>"
operand_expr := expr / "_" / expr "=>" expr / expr "=>" "_"
reg_operand := dir_spec "(" reg_spec ")" operand_expr
operand := reg_operand / "const" const_expr / "sym" path
option := "pure" / "nomem" / "readonly" / "preserves_flags" / "noreturn" / "nostack" / "att_syntax"
options := "options(" option *["," option] [","] ")"
asm := "asm!(" format_string *("," format_string) *("," [ident "="] operand) ["," options] [","] ")"
```

The macro will initially be supported only on ARM, AArch64, Hexagon, PowerPC, x86, x86-64 and RISC-V targets. Support for more targets may be added in the future. The compiler will emit an error if `asm!` is used on an unsupported target.

[format-syntax]: https://doc.rust-lang.org/std/fmt/#syntax

## Template string arguments

The assembler template uses the same syntax as [format strings][format-syntax] (i.e. placeholders are specified by curly braces). The corresponding arguments are accessed in order, by index, or by name. However, implicit named arguments (introduced by [RFC #2795][rfc-2795]) are not supported.

An `asm!` invocation may have one or more template string arguments; an `asm!` with multiple template string arguments is treated as if all the strings were concatenated with a `\n` between them. The expected usage is for each template string argument to correspond to a line of assembly code. All template string arguments must appear before any other arguments.

As with format strings, named arguments must appear after positional arguments. Explicit register operands must appear at the end of the operand list, after named arguments if any.

Explicit register operands cannot be used by placeholders in the template string. All other named and positional operands must appear at least once in the template string, otherwise a compiler error is generated.

The exact assembly code syntax is target-specific and opaque to the compiler except for the way operands are substituted into the template string to form the code passed to the assembler.

The 5 targets specified in this RFC (x86, ARM, AArch64, RISC-V, Hexagon) all use the assembly code syntax of the GNU assembler (GAS). On x86, the `.intel_syntax noprefix` mode of GAS is used by default. On ARM, the `.syntax unified` mode is used. These targets impose an additional restriction on the assembly code: any assembler state (e.g. the current section which can be changed with `.section`) must be restored to its original value at the end of the asm string. Assembly code that does not conform to the GAS syntax will result in assembler-specific behavior.

[rfc-2795]: https://github.com/rust-lang/rfcs/pull/2795

## Operand type

Several types of operands are supported:

* `in(<reg>) <expr>`
  - `<reg>` can refer to a register class or an explicit register. The allocated register name is substituted into the asm template string.
  - The allocated register will contain the value of `<expr>` at the start of the asm code.
  - The allocated register must contain the same value at the end of the asm code (except if a `lateout` is allocated to the same register).
* `out(<reg>) <expr>`
  - `<reg>` can refer to a register class or an explicit register. The allocated register name is substituted into the asm template string.
  - The allocated register will contain an undefined value at the start of the asm code.
  - `<expr>` must be a (possibly uninitialized) place expression, to which the contents of the allocated register is written to at the end of the asm code.
  - An underscore (`_`) may be specified instead of an expression, which will cause the contents of the register to be discarded at the end of the asm code (effectively acting as a clobber).
* `lateout(<reg>) <expr>`
  - Identical to `out` except that the register allocator can reuse a register allocated to an `in`.
  - You should only write to the register after all inputs are read, otherwise you may clobber an input.
* `inout(<reg>) <expr>`
  - `<reg>` can refer to a register class or an explicit register. The allocated register name is substituted into the asm template string.
  - The allocated register will contain the value of `<expr>` at the start of the asm code.
  - `<expr>` must be a mutable initialized place expression, to which the contents of the allocated register is written to at the end of the asm code.
* `inout(<reg>) <in expr> => <out expr>`
  - Same as `inout` except that the initial value of the register is taken from the value of `<in expr>`.
  - `<out expr>` must be a (possibly uninitialized) place expression, to which the contents of the allocated register is written to at the end of the asm code.
  - An underscore (`_`) may be specified instead of an expression for `<out expr>`, which will cause the contents of the register to be discarded at the end of the asm code (effectively acting as a clobber).
  - `<in expr>` and `<out expr>` may have different types.
* `inlateout(<reg>) <expr>` / `inlateout(<reg>) <in expr> => <out expr>`
  - Identical to `inout` except that the register allocator can reuse a register allocated to an `in` (this can happen if the compiler knows the `in` has the same initial value as the `inlateout`).
  - You should only write to the register after all inputs are read, otherwise you may clobber an input.
* `const <expr>`
  - `<expr>` must be an integer constant expression.
  - The value of the expression is formatted as a string and substituted directly into the asm template string.
* `sym <path>`
  - `<path>` must refer to a `fn` or `static`.
  - A mangled symbol name referring to the item is substituted into the asm template string.
  - The substituted string does not include any modifiers (e.g. GOT, PLT, relocations, etc).
  - `<path>` is allowed to point to a `#[thread_local]` static, in which case the asm code can combine the symbol with relocations (e.g. `@plt`, `@TPOFF`) to read from thread-local data.

Operand expressions are evaluated from left to right, just like function call arguments. After the `asm!` has executed, outputs are written to in left to right order. This is significant if two outputs point to the same place: that place will contain the value of the rightmost output.

## Register operands

Input and output operands can be specified either as an explicit register or as a register class from which the register allocator can select a register. Explicit registers are specified as string literals (e.g. `"eax"`) while register classes are specified as identifiers (e.g. `reg`). Using string literals for register names enables support for architectures that use special characters in register names, such as MIPS (`$0`, `$1`, etc).

Note that explicit registers treat register aliases (e.g. `r14` vs `lr` on ARM) and smaller views of a register (e.g. `eax` vs `rax`) as equivalent to the base register. It is a compile-time error to use the same explicit register for two input operands or two output operands. Additionally, it is also a compile-time error to use overlapping registers (e.g. ARM VFP) in input operands or in output operands.

Only the following types are allowed as operands for inline assembly:
- Integers (signed and unsigned)
- Floating-point numbers
- Pointers (thin only)
- Function pointers
- SIMD vectors (structs defined with `#[repr(simd)]` and which implement `Copy`). This includes architecture-specific vector types defined in `std::arch` such as `__m128` (x86) or `int8x16_t` (ARM).

Here is the list of currently supported register classes:

| Architecture | Register class | Registers | LLVM constraint code |
| ------------ | -------------- | --------- | -------------------- |
| x86 | `reg` | `ax`, `bx`, `cx`, `dx`, `si`, `di`, `bp`, `r[8-15]` (x86-64 only) | `r` |
| x86 | `reg_abcd` | `ax`, `bx`, `cx`, `dx` | `Q` |
| x86-32 | `reg_byte` | `al`, `bl`, `cl`, `dl`, `ah`, `bh`, `ch`, `dh` | `q` |
| x86-64 | `reg_byte`\* | `al`, `bl`, `cl`, `dl`, `sil`, `dil`, `bpl`, `r[8-15]b` | `q` |
| x86 | `xmm_reg` | `xmm[0-7]` (x86) `xmm[0-15]` (x86-64) | `x` |
| x86 | `ymm_reg` | `ymm[0-7]` (x86) `ymm[0-15]` (x86-64) | `x` |
| x86 | `zmm_reg` | `zmm[0-7]` (x86) `zmm[0-31]` (x86-64) | `v` |
| x86 | `kreg` | `k[1-7]` | `Yk` |
| AArch64 | `reg` | `x[0-30]` | `r` |
| AArch64 | `vreg` | `v[0-31]` | `w` |
| AArch64 | `vreg_low16` | `v[0-15]` | `x` |
| ARM | `reg` | `r[0-12]`, `r14` | `r` |
| ARM (Thumb) | `reg_thumb` | `r[0-r7]` | `l` |
| ARM (ARM) | `reg_thumb` | `r[0-r12]`, `r14` | `l` |
| ARM | `sreg` | `s[0-31]` | `t` |
| ARM | `sreg_low16` | `s[0-15]` | `x` |
| ARM | `dreg` | `d[0-31]` | `w` |
| ARM | `dreg_low16` | `d[0-15]` | `t` |
| ARM | `dreg_low8` | `d[0-8]` | `x` |
| ARM | `qreg` | `q[0-15]` | `w` |
| ARM | `qreg_low8` | `q[0-7]` | `t` |
| ARM | `qreg_low4` | `q[0-3]` | `x` |
| MIPS | `reg` | `$[2-25]` | `r` |
| MIPS | `freg` | `$f[0-31]` | `f` |
| NVPTX | `reg16` | None\* | `h` |
| NVPTX | `reg32` | None\* | `r` |
| NVPTX | `reg64` | None\* | `l` |
| RISC-V | `reg` | `x1`, `x[5-7]`, `x[9-15]`, `x[16-31]` (non-RV32E) | `r` |
| RISC-V | `freg` | `f[0-31]` | `f` |
| Hexagon | `reg` | `r[0-28]` | `r` |
| PowerPC | `reg` | `r[0-31]` | `r` |
| PowerPC | `reg_nonzero` | | `r[1-31]` | `b` |
| PowerPC | `freg` | `f[0-31]` | `f` |
| wasm32 | `local` | None\* | `r` |

> **Note**: On x86 we treat `reg_byte` differently from `reg` because the compiler can allocate `al` and `ah` separately whereas `reg` reserves the whole register.
>
> Note #2: On x86-64 the high byte registers (e.g. `ah`) are not available in the `reg_byte` register class.
>
> Note #3: NVPTX doesn't have a fixed register set, so named registers are not supported.
>
> Note #4: WebAssembly doesn't have registers, so named registers are not supported.

Additional register classes may be added in the future based on demand (e.g. MMX, x87, etc).

Each register class has constraints on which value types they can be used with. This is necessary because the way a value is loaded into a register depends on its type. For example, on big-endian systems, loading a `i32x4` and a `i8x16` into a SIMD register may result in different register contents even if the byte-wise memory representation of both values is identical. The availability of supported types for a particular register class may depend on what target features are currently enabled.

| Architecture | Register class | Target feature | Allowed types |
| ------------ | -------------- | -------------- | ------------- |
| x86-32 | `reg` | None | `i16`, `i32`, `f32` |
| x86-64 | `reg` | None | `i16`, `i32`, `f32`, `i64`, `f64` |
| x86 | `reg_byte` | None | `i8` |
| x86 | `xmm_reg` | `sse` | `i32`, `f32`, `i64`, `f64`, <br> `i8x16`, `i16x8`, `i32x4`, `i64x2`, `f32x4`, `f64x2` |
| x86 | `ymm_reg` | `avx` | `i32`, `f32`, `i64`, `f64`, <br> `i8x16`, `i16x8`, `i32x4`, `i64x2`, `f32x4`, `f64x2` <br> `i8x32`, `i16x16`, `i32x8`, `i64x4`, `f32x8`, `f64x4` |
| x86 | `zmm_reg` | `avx512f` | `i32`, `f32`, `i64`, `f64`, <br> `i8x16`, `i16x8`, `i32x4`, `i64x2`, `f32x4`, `f64x2` <br> `i8x32`, `i16x16`, `i32x8`, `i64x4`, `f32x8`, `f64x4` <br> `i8x64`, `i16x32`, `i32x16`, `i64x8`, `f32x16`, `f64x8` |
| x86 | `kreg` | `axv512f` | `i8`, `i16` |
| x86 | `kreg` | `axv512bw` | `i32`, `i64` |
| AArch64 | `reg` | None | `i8`, `i16`, `i32`, `f32`, `i64`, `f64` |
| AArch64 | `vreg` | `fp` | `i8`, `i16`, `i32`, `f32`, `i64`, `f64`, <br> `i8x8`, `i16x4`, `i32x2`, `i64x1`, `f32x2`, `f64x1`, <br> `i8x16`, `i16x8`, `i32x4`, `i64x2`, `f32x4`, `f64x2` |
| ARM | `reg` | None | `i8`, `i16`, `i32`, `f32` |
| ARM | `sreg` | `vfp2` | `i32`, `f32` |
| ARM | `dreg` | `vfp2` | `i64`, `f64`, `i8x8`, `i16x4`, `i32x2`, `i64x1`, `f32x2` |
| ARM | `qreg` | `neon` | `i8x16`, `i16x8`, `i32x4`, `i64x2`, `f32x4` |
| MIPS32 | `reg` | None | `i8`, `i16`, `i32`, `f32` |
| MIPS32 | `freg` | None | `f32`, `f64` |
| MIPS64 | `reg` | None | `i8`, `i16`, `i32`, `i64`, `f32`, `f64` |
| MIPS64 | `freg` | None | `f32`, `f64` |
| NVPTX | `reg16` | None | `i8`, `i16` |
| NVPTX | `reg32` | None | `i8`, `i16`, `i32`, `f32` |
| NVPTX | `reg64` | None | `i8`, `i16`, `i32`, `f32`, `i64`, `f64` |
| RISC-V32 | `reg` | None | `i8`, `i16`, `i32`, `f32` |
| RISC-V64 | `reg` | None | `i8`, `i16`, `i32`, `f32`, `i64`, `f64` |
| RISC-V | `freg` | `f` | `f32` |
| RISC-V | `freg` | `d` | `f64` |
| Hexagon | `reg` | None | `i8`, `i16`, `i32`, `f32` |
| PowerPC | `reg` | None | `i8`, `i16`, `i32` |
| PowerPC | `reg_nonzero` | None | `i8`, `i16`, `i32` |
| PowerPC | `freg` | None | `f32`, `f64` |
| wasm32 | `local` | None | `i8` `i16` `i32` `i64` `f32` `f64` |

> **Note**: For the purposes of the above table pointers, function pointers and `isize`/`usize` are treated as the equivalent integer type (`i16`/`i32`/`i64` depending on the target).

If a value is of a smaller size than the register it is allocated in then the upper bits of that register will have an undefined value for inputs and will be ignored for outputs. The only exception is the `freg` register class on RISC-V where `f32` values are NaN-boxed in a `f64` as required by the RISC-V architecture.

When separate input and output expressions are specified for an `inout` operand, both expressions must have the same type. The only exception is if both operands are pointers or integers, in which case they are only required to have the same size. This restriction exists because the register allocators in LLVM and GCC sometimes cannot handle tied operands with different types.

## Register names

Some registers have multiple names. These are all treated by the compiler as identical to the base register name. Here is the list of all supported register aliases:

| Architecture | Base register | Aliases |
| ------------ | ------------- | ------- |
| x86 | `ax` | `eax`, `rax` |
| x86 | `bx` | `ebx`, `rbx` |
| x86 | `cx` | `ecx`, `rcx` |
| x86 | `dx` | `edx`, `rdx` |
| x86 | `si` | `esi`, `rsi` |
| x86 | `di` | `edi`, `rdi` |
| x86 | `bp` | `bpl`, `ebp`, `rbp` |
| x86 | `sp` | `spl`, `esp`, `rsp` |
| x86 | `ip` | `eip`, `rip` |
| x86 | `st(0)` | `st` |
| x86 | `r[8-15]` | `r[8-15]b`, `r[8-15]w`, `r[8-15]d` |
| x86 | `xmm[0-31]` | `ymm[0-31]`, `zmm[0-31]` |
| AArch64 | `x[0-30]` | `w[0-30]` |
| AArch64 | `x29` | `fp` |
| AArch64 | `x30` | `lr` |
| AArch64 | `sp` | `wsp` |
| AArch64 | `xzr` | `wzr` |
| AArch64 | `v[0-31]` | `b[0-31]`, `h[0-31]`, `s[0-31]`, `d[0-31]`, `q[0-31]` |
| ARM | `r[0-3]` | `a[1-4]` |
| ARM | `r[4-9]` | `v[1-6]` |
| ARM | `r9` | `rfp` |
| ARM | `r10` | `sl` |
| ARM | `r11` | `fp` |
| ARM | `r12` | `ip` |
| ARM | `r13` | `sp` |
| ARM | `r14` | `lr` |
| ARM | `r15` | `pc` |
| RISC-V | `x0` | `zero` |
| RISC-V | `x1` | `ra` |
| RISC-V | `x2` | `sp` |
| RISC-V | `x3` | `gp` |
| RISC-V | `x4` | `tp` |
| RISC-V | `x[5-7]` | `t[0-2]` |
| RISC-V | `x8` | `fp`, `s0` |
| RISC-V | `x9` | `s1` |
| RISC-V | `x[10-17]` | `a[0-7]` |
| RISC-V | `x[18-27]` | `s[2-11]` |
| RISC-V | `x[28-31]` | `t[3-6]` |
| RISC-V | `f[0-7]` | `ft[0-7]` |
| RISC-V | `f[8-9]` | `fs[0-1]` |
| RISC-V | `f[10-17]` | `fa[0-7]` |
| RISC-V | `f[18-27]` | `fs[2-11]` |
| RISC-V | `f[28-31]` | `ft[8-11]` |
| Hexagon | `r29` | `sp` |
| Hexagon | `r30` | `fr` |
| Hexagon | `r31` | `lr` |

Some registers cannot be used for input or output operands:

| Architecture | Unsupported register | Reason |
| ------------ | -------------------- | ------ |
| All | `sp` | The stack pointer must be restored to its original value at the end of an asm code block. |
| All | `bp` (x86), `x29` (AArch64), `x8` (RISC-V), `fr` (Hexagon), `$fp` (MIPS) | The frame pointer cannot be used as an input or output. |
| ARM | `r7` or `r11` | On ARM the frame pointer can be either `r7` or `r11` depending on the target. The frame pointer cannot be used as an input or output. |
| All | `si` (x86-32), `bx` (x86-64), `r6` (ARM), `x19` (AArch64), `r19` (Hexagon), `x9` (RISC-V) | This is used internally by LLVM as a "base pointer" for functions with complex stack frames. |
| x86 | `k0` | This is a constant zero register which can't be modified. |
| x86 | `ip` | This is the program counter, not a real register. |
| x86 | `mm[0-7]` | MMX registers are not currently supported (but may be in the future). |
| x86 | `st([0-7])` | x87 registers are not currently supported (but may be in the future). |
| AArch64 | `xzr` | This is a constant zero register which can't be modified. |
| ARM | `pc` | This is the program counter, not a real register. |
| ARM | `r9` | This is a reserved register on some ARM targets. |
| MIPS | `$0` or `$zero` | This is a constant zero register which can't be modified. |
| MIPS | `$1` or `$at` | Reserved for assembler. |
| MIPS | `$26`/`$k0`, `$27`/`$k1` | OS-reserved registers. |
| MIPS | `$28`/`$gp` | Global pointer cannot be used as inputs or outputs. |
| MIPS | `$ra` | Return address cannot be used as inputs or outputs. |
| RISC-V | `x0` | This is a constant zero register which can't be modified. |
| RISC-V | `gp`, `tp` | These registers are reserved and cannot be used as inputs or outputs. |
| Hexagon | `lr` | This is the link register which cannot be used as an input or output. |

In some cases LLVM will allocate a "reserved register" for `reg` operands even though this register cannot be explicitly specified. Assembly code making use of reserved registers should be careful since `reg` operands may alias with those registers. Reserved registers are the frame pointer and base pointer
- The frame pointer and LLVM base pointer on all architectures.
- `r9` on ARM.
- `x18` on AArch64.

## Template modifiers

The placeholders can be augmented by modifiers which are specified after the `:` in the curly braces. These modifiers do not affect register allocation, but change the way operands are formatted when inserted into the template string. Only one modifier is allowed per template placeholder.

The supported modifiers are a subset of LLVM's (and GCC's) [asm template argument modifiers][llvm-argmod], but do not use the same letter codes.

| Architecture | Register class | Modifier | Example output | LLVM modifier |
| ------------ | -------------- | -------- | -------------- | ------------- |
| x86-32 | `reg` | None | `eax` | `k` |
| x86-64 | `reg` | None | `rax` | `q` |
| x86-32 | `reg_abcd` | `l` | `al` | `b` |
| x86-64 | `reg` | `l` | `al` | `b` |
| x86 | `reg_abcd` | `h` | `ah` | `h` |
| x86 | `reg` | `x` | `ax` | `w` |
| x86 | `reg` | `e` | `eax` | `k` |
| x86-64 | `reg` | `r` | `rax` | `q` |
| x86 | `reg_byte` | None | `al` / `ah` | None |
| x86 | `xmm_reg` | None | `xmm0` | `x` |
| x86 | `ymm_reg` | None | `ymm0` | `t` |
| x86 | `zmm_reg` | None | `zmm0` | `g` |
| x86 | `*mm_reg` | `x` | `xmm0` | `x` |
| x86 | `*mm_reg` | `y` | `ymm0` | `t` |
| x86 | `*mm_reg` | `z` | `zmm0` | `g` |
| x86 | `kreg` | None | `k1` | None |
| AArch64 | `reg` | None | `x0` | `x` |
| AArch64 | `reg` | `w` | `w0` | `w` |
| AArch64 | `reg` | `x` | `x0` | `x` |
| AArch64 | `vreg` | None | `v0` | None |
| AArch64 | `vreg` | `v` | `v0` | None |
| AArch64 | `vreg` | `b` | `b0` | `b` |
| AArch64 | `vreg` | `h` | `h0` | `h` |
| AArch64 | `vreg` | `s` | `s0` | `s` |
| AArch64 | `vreg` | `d` | `d0` | `d` |
| AArch64 | `vreg` | `q` | `q0` | `q` |
| ARM | `reg` | None | `r0` | None |
| ARM | `sreg` | None | `s0` | None |
| ARM | `dreg` | None | `d0` | `P` |
| ARM | `qreg` | None | `q0` | `q` |
| ARM | `qreg` | `e` / `f` | `d0` / `d1` | `e` / `f` |
| MIPS | `reg` | None | `$2` | None |
| MIPS | `freg` | None | `$f0` | None |
| NVPTX | `reg16` | None | `rs0` | None |
| NVPTX | `reg32` | None | `r0` | None |
| NVPTX | `reg64` | None | `rd0` | None |
| RISC-V | `reg` | None | `x1` | None |
| RISC-V | `freg` | None | `f0` | None |
| Hexagon | `reg` | None | `r0` | None |
| PowerPC | `reg` | None | `0` | None |
| PowerPC | `reg_nonzero` | None | `3` | `b` |
| PowerPC | `freg` | None | `0` | None |

> Notes:
> - on ARM `e` / `f`: this prints the low or high doubleword register name of a NEON quad (128-bit) register.
> - on x86: our behavior for `reg` with no modifiers differs from what GCC does. GCC will infer the modifier based on the operand value type, while we default to the full register size.
> - on x86 `xmm_reg`: the `x`, `t` and `g` LLVM modifiers are not yet implemented in LLVM (they are supported by GCC only), but this should be a simple change.

As stated in the previous section, passing an input value smaller than the register width will result in the upper bits of the register containing undefined values. This is not a problem if the inline asm only accesses the lower bits of the register, which can be done by using a template modifier to use a subregister name in the asm code (e.g. `ax` instead of `rax`). Since this an easy pitfall, the compiler will suggest a template modifier to use where appropriate given the input type. If all references to an operand already have modifiers then the warning is suppressed for that operand.

[llvm-argmod]: http://llvm.org/docs/LangRef.html#asm-template-argument-modifiers

## Options

Flags are used to further influence the behavior of the inline assembly block.
Currently the following options are defined:
- `pure`: The `asm` block has no side effects, and its outputs depend only on its direct inputs (i.e. the values themselves, not what they point to) or values read from memory (unless the `nomem` options is also set). This allows the compiler to execute the `asm` block fewer times than specified in the program (e.g. by hoisting it out of a loop) or even eliminate it entirely if the outputs are not used.
- `nomem`: The `asm` blocks does not read or write to any memory. This allows the compiler to cache the values of modified global variables in registers across the `asm` block since it knows that they are not read or written to by the `asm`.
- `readonly`: The `asm` block does not write to any memory. This allows the compiler to cache the values of unmodified global variables in registers across the `asm` block since it knows that they are not written to by the `asm`.
- `preserves_flags`: The `asm` block does not modify the flags register (defined in the rules below). This allows the compiler to avoid recomputing the condition flags after the `asm` block.
- `noreturn`: The `asm` block never returns, and its return type is defined as `!` (never). Behavior is undefined if execution falls through past the end of the asm code. A `noreturn` asm block behaves just like a function which doesn't return; notably, local variables in scope are not dropped before it is invoked.
- `nostack`: The `asm` block does not push data to the stack, or write to the stack red-zone (if supported by the target). If this option is *not* used then the stack pointer is guaranteed to be suitably aligned (according to the target ABI) for a function call.
- `att_syntax`: This option is only valid on x86, and causes the assembler to use the `.att_syntax prefix` mode of the GNU assembler. Register operands are substituted in with a leading `%`.

The compiler performs some additional checks on options:
- The `nomem` and `readonly` options are mutually exclusive: it is a compile-time error to specify both.
- The `pure` option must be combined with either the `nomem` or `readonly` options, otherwise a compile-time error is emitted.
- It is a compile-time error to specify `pure` on an asm block with no outputs or only discarded outputs (`_`).
- It is a compile-time error to specify `noreturn` on an asm block with outputs.

## Rules for inline assembly

- Any registers not specified as inputs will contain an undefined value on entry to the asm block.
  - An "undefined value" in the context of inline assembly means that the register can (non-deterministically) have any one of the possible values allowed by the architecture. Notably it is not the same as an LLVM `undef` which can have a different value every time you read it (since such a concept does not exist in assembly code).
- Any registers not specified as outputs must have the same value upon exiting the asm block as they had on entry, otherwise behavior is undefined.
  - This only applies to registers which can be specified as an input or output. Other registers follow target-specific rules.
  - Note that a `lateout` may be allocated to the same register as an `in`, in which case this rule does not apply. Code should not rely on this however since it depends on the results of register allocation.
- Behavior is undefined if execution unwinds out of an asm block.
  - This also applies if the assembly code calls a function which then unwinds.
- The set of memory locations that assembly code is allowed the read and write are the same as those allowed for an FFI function.
  - Refer to the unsafe code guidelines for the exact rules.
  - If the `readonly` option is set, then only memory reads are allowed.
  - If the `nomem` option is set then no reads or writes to memory are allowed.
  - These rules do not apply to memory which is private to the asm code, such as stack space allocated within the asm block.
- The compiler cannot assume that the instructions in the asm are the ones that will actually end up executed.
  - This effectively means that the compiler must treat the `asm!` as a black box and only take the interface specification into account, not the instructions themselves.
  - Runtime code patching is allowed, via target-specific mechanisms (outside the scope of this RFC).
- Unless the `nostack` option is set, asm code is allowed to use stack space below the stack pointer.
  - On entry to the asm block the stack pointer is guaranteed to be suitably aligned (according to the target ABI) for a function call.
  - You are responsible for making sure you don't overflow the stack (e.g. use stack probing to ensure you hit a guard page).
  - You should adjust the stack pointer when allocating stack memory as required by the target ABI.
  - The stack pointer must be restored to its original value before leaving the asm block.
- If the `noreturn` option is set then behavior is undefined if execution falls through to the end of the asm block.
- If the `pure` option is set then behavior is undefined if the `asm` has side-effects other than its direct outputs. Behavior is also undefined if two executions of the `asm` code with the same inputs result in different outputs.
  - When used with the `nomem` option, "inputs" are just the direct inputs of the `asm!`.
  - When used with the `readonly` option, "inputs" comprise the direct inputs of the `asm!` and any memory that the `asm!` block is allowed to read.
- These flags registers must be restored upon exiting the asm block if the `preserves_flags` option is set:
  - x86
    - Status flags in `EFLAGS` (CF, PF, AF, ZF, SF, OF).
    - Floating-point status word (all).
    - Floating-point exception flags in `MXCSR` (PE, UE, OE, ZE, DE, IE).
  - ARM
    - Condition flags in `CPSR` (N, Z, C, V)
    - Saturation flag in `CPSR` (Q)
    - Greater than or equal flags in `CPSR` (GE).
    - Condition flags in `FPSCR` (N, Z, C, V)
    - Saturation flag in `FPSCR` (QC)
    - Floating-point exception flags in `FPSCR` (IDC, IXC, UFC, OFC, DZC, IOC).
  - AArch64
    - Condition flags (`NZCV` register).
    - Floating-point status (`FPSR` register).
  - RISC-V
    - Floating-point exception flags in `fcsr` (`fflags`).
- On x86, the direction flag (DF in `EFLAGS`) is clear on entry to an asm block and must be clear on exit.
  - Behavior is undefined if the direction flag is set on exiting an asm block.
- The requirement of restoring the stack pointer and non-output registers to their original value only applies when exiting an `asm!` block.
  - This means that `asm!` blocks that never return (even if not marked `noreturn`) don't need to preserve these registers.
  - When returning to a different `asm!` block than you entered (e.g. for context switching), these registers must contain the value they had upon entering the `asm!` block that you are *exiting*.
    - You cannot exit an `asm!` block that has not been entered. Neither can you exit an `asm!` block that has already been exited.
    - You are responsible for switching any target-specific state (e.g. thread-local storage, stack bounds).
    - The set of memory locations that you may access is the intersection of those allowed by the `asm!` blocks you entered and exited.
- You cannot assume that an `asm!` block will appear exactly once in the output binary. The compiler is allowed to instantiate multiple copies of the `asm!` block, for example when the function containing it is inlined in multiple places.

> **Note**: As a general rule, the flags covered by `preserves_flags` are those which are *not* preserved when performing a function call.
"##,
    },
    Lint {
        label: "auto_traits",
        description: r##"# `auto_traits`

The tracking issue for this feature is [#13231]

[#13231]: https://github.com/rust-lang/rust/issues/13231

----

The `auto_traits` feature gate allows you to define auto traits.

Auto traits, like [`Send`] or [`Sync`] in the standard library, are marker traits
that are automatically implemented for every type, unless the type, or a type it contains,
has explicitly opted out via a negative impl. (Negative impls are separately controlled
by the `negative_impls` feature.)

[`Send`]: https://doc.rust-lang.org/std/marker/trait.Send.html
[`Sync`]: https://doc.rust-lang.org/std/marker/trait.Sync.html

```rust,ignore (partial-example)
impl !Trait for Type {}
```

Example:

```rust
#![feature(negative_impls)]
#![feature(auto_traits)]

auto trait Valid {}

struct True;
struct False;

impl !Valid for False {}

struct MaybeValid<T>(T);

fn must_be_valid<T: Valid>(_t: T) { }

fn main() {
    // works
    must_be_valid( MaybeValid(True) );

    // compiler error - trait bound not satisfied
    // must_be_valid( MaybeValid(False) );
}
```

## Automatic trait implementations

When a type is declared as an `auto trait`, we will automatically
create impls for every struct/enum/union, unless an explicit impl is
provided. These automatic impls contain a where clause for each field
of the form `T: AutoTrait`, where `T` is the type of the field and
`AutoTrait` is the auto trait in question. As an example, consider the
struct `List` and the auto trait `Send`:

```rust
struct List<T> {
  data: T,
  next: Option<Box<List<T>>>,
}
```

Presuming that there is no explicit impl of `Send` for `List`, the
compiler will supply an automatic impl of the form:

```rust
struct List<T> {
  data: T,
  next: Option<Box<List<T>>>,
}

unsafe impl<T> Send for List<T>
where
  T: Send, // from the field `data`
  Option<Box<List<T>>>: Send, // from the field `next`
{ }
```

Explicit impls may be either positive or negative. They take the form:

```rust,ignore (partial-example)
impl<...> AutoTrait for StructName<..> { }
impl<...> !AutoTrait for StructName<..> { }
```

## Coinduction: Auto traits permit cyclic matching

Unlike ordinary trait matching, auto traits are **coinductive**. This
means, in short, that cycles which occur in trait matching are
considered ok. As an example, consider the recursive struct `List`
introduced in the previous section. In attempting to determine whether
`List: Send`, we would wind up in a cycle: to apply the impl, we must
show that `Option<Box<List>>: Send`, which will in turn require
`Box<List>: Send` and then finally `List: Send` again. Under ordinary
trait matching, this cycle would be an error, but for an auto trait it
is considered a successful match.

## Items

Auto traits cannot have any trait items, such as methods or associated types. This ensures that we can generate default implementations.

## Supertraits

Auto traits cannot have supertraits. This is for soundness reasons, as the interaction of coinduction with implied bounds is difficult to reconcile.
"##,
    },
    Lint {
        label: "box_patterns",
        description: r##"# `box_patterns`

The tracking issue for this feature is: [#29641]

[#29641]: https://github.com/rust-lang/rust/issues/29641

See also [`box_syntax`](box-syntax.md)

------------------------

Box patterns let you match on `Box<T>`s:


```rust
#![feature(box_patterns)]

fn main() {
    let b = Some(Box::new(5));
    match b {
        Some(box n) if n < 0 => {
            println!("Box contains negative number {}", n);
        },
        Some(box n) if n >= 0 => {
            println!("Box contains non-negative number {}", n);
        },
        None => {
            println!("No box");
        },
        _ => unreachable!()
    }
}
```
"##,
    },
    Lint {
        label: "box_syntax",
        description: r##"# `box_syntax`

The tracking issue for this feature is: [#49733]

[#49733]: https://github.com/rust-lang/rust/issues/49733

See also [`box_patterns`](box-patterns.md)

------------------------

Currently the only stable way to create a `Box` is via the `Box::new` method.
Also it is not possible in stable Rust to destructure a `Box` in a match
pattern. The unstable `box` keyword can be used to create a `Box`. An example
usage would be:

```rust
#![feature(box_syntax)]

fn main() {
    let b = box 5;
}
```
"##,
    },
    Lint {
        label: "c_unwind",
        description: r##"# `c_unwind`

The tracking issue for this feature is: [#74990]

[#74990]: https://github.com/rust-lang/rust/issues/74990

------------------------

Introduces four new ABI strings: "C-unwind", "stdcall-unwind",
"thiscall-unwind", and "system-unwind". These enable unwinding from other
languages (such as C++) into Rust frames and from Rust into other languages.

See [RFC 2945] for more information.

[RFC 2945]: https://github.com/rust-lang/rfcs/blob/master/text/2945-c-unwind-abi.md
"##,
    },
    Lint {
        label: "c_variadic",
        description: r##"# `c_variadic`

The tracking issue for this feature is: [#44930]

[#44930]: https://github.com/rust-lang/rust/issues/44930

------------------------

The `c_variadic` language feature enables C-variadic functions to be
defined in Rust. The may be called both from within Rust and via FFI.

## Examples

```rust
#![feature(c_variadic)]

pub unsafe extern "C" fn add(n: usize, mut args: ...) -> usize {
    let mut sum = 0;
    for _ in 0..n {
        sum += args.arg::<usize>();
    }
    sum
}
```
"##,
    },
    Lint {
        label: "c_variadic",
        description: r##"# `c_variadic`

The tracking issue for this feature is: [#44930]

[#44930]: https://github.com/rust-lang/rust/issues/44930

------------------------

The `c_variadic` library feature exposes the `VaList` structure,
Rust's analogue of C's `va_list` type.

## Examples

```rust
#![feature(c_variadic)]

use std::ffi::VaList;

pub unsafe extern "C" fn vadd(n: usize, mut args: VaList) -> usize {
    let mut sum = 0;
    for _ in 0..n {
        sum += args.arg::<usize>();
    }
    sum
}
```
"##,
    },
    Lint {
        label: "c_void_variant",
        description: r##"# `c_void_variant`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "cfg_panic",
        description: r##"# `cfg_panic`

The tracking issue for this feature is: [#77443]

[#77443]: https://github.com/rust-lang/rust/issues/77443

------------------------

The `cfg_panic` feature makes it possible to execute different code
depending on the panic strategy.

Possible values at the moment are `"unwind"` or `"abort"`, although
it is possible that new panic strategies may be added to Rust in the
future.

## Examples

```rust
#![feature(cfg_panic)]

#[cfg(panic = "unwind")]
fn a() {
    // ...
}

#[cfg(not(panic = "unwind"))]
fn a() {
    // ...
}

fn b() {
    if cfg!(panic = "abort") {
        // ...
    } else {
        // ...
    }
}
```
"##,
    },
    Lint {
        label: "cfg_sanitize",
        description: r##"# `cfg_sanitize`

The tracking issue for this feature is: [#39699]

[#39699]: https://github.com/rust-lang/rust/issues/39699

------------------------

The `cfg_sanitize` feature makes it possible to execute different code
depending on whether a particular sanitizer is enabled or not.

## Examples

```rust
#![feature(cfg_sanitize)]

#[cfg(sanitize = "thread")]
fn a() {
    // ...
}

#[cfg(not(sanitize = "thread"))]
fn a() {
    // ...
}

fn b() {
    if cfg!(sanitize = "leak") {
        // ...
    } else {
        // ...
    }
}
```
"##,
    },
    Lint {
        label: "cfg_version",
        description: r##"# `cfg_version`

The tracking issue for this feature is: [#64796]

[#64796]: https://github.com/rust-lang/rust/issues/64796

------------------------

The `cfg_version` feature makes it possible to execute different code
depending on the compiler version. It will return true if the compiler
version is greater than or equal to the specified version.

## Examples

```rust
#![feature(cfg_version)]

#[cfg(version("1.42"))] // 1.42 and above
fn a() {
    // ...
}

#[cfg(not(version("1.42")))] // 1.41 and below
fn a() {
    // ...
}

fn b() {
    if cfg!(version("1.42")) {
        // ...
    } else {
        // ...
    }
}
```
"##,
    },
    Lint {
        label: "char_error_internals",
        description: r##"# `char_error_internals`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "cmse_nonsecure_entry",
        description: r##"# `cmse_nonsecure_entry`

The tracking issue for this feature is: [#75835]

[#75835]: https://github.com/rust-lang/rust/issues/75835

------------------------

The [TrustZone-M
feature](https://developer.arm.com/documentation/100690/latest/) is available
for targets with the Armv8-M architecture profile (`thumbv8m` in their target
name).
LLVM, the Rust compiler and the linker are providing
[support](https://developer.arm.com/documentation/ecm0359818/latest/) for the
TrustZone-M feature.

One of the things provided, with this unstable feature, is the
`cmse_nonsecure_entry` attribute.  This attribute marks a Secure function as an
entry function (see [section
5.4](https://developer.arm.com/documentation/ecm0359818/latest/) for details).
With this attribute, the compiler will do the following:
* add a special symbol on the function which is the `__acle_se_` prefix and the
  standard function name
* constrain the number of parameters to avoid using the Non-Secure stack
* before returning from the function, clear registers that might contain Secure
  information
* use the `BXNS` instruction to return

Because the stack can not be used to pass parameters, there will be compilation
errors if:
* the total size of all parameters is too big (for example more than four 32
  bits integers)
* the entry function is not using a C ABI

The special symbol `__acle_se_` will be used by the linker to generate a secure
gateway veneer.

<!-- NOTE(ignore) this example is specific to thumbv8m targets -->

``` rust,ignore
#![feature(cmse_nonsecure_entry)]

#[no_mangle]
#[cmse_nonsecure_entry]
pub extern "C" fn entry_function(input: u32) -> u32 {
    input + 6
}
```

``` text
$ rustc --emit obj --crate-type lib --target thumbv8m.main-none-eabi function.rs
$ arm-none-eabi-objdump -D function.o

00000000 <entry_function>:
   0:   b580            push    {r7, lr}
   2:   466f            mov     r7, sp
   4:   b082            sub     sp, #8
   6:   9001            str     r0, [sp, #4]
   8:   1d81            adds    r1, r0, #6
   a:   460a            mov     r2, r1
   c:   4281            cmp     r1, r0
   e:   9200            str     r2, [sp, #0]
  10:   d30b            bcc.n   2a <entry_function+0x2a>
  12:   e7ff            b.n     14 <entry_function+0x14>
  14:   9800            ldr     r0, [sp, #0]
  16:   b002            add     sp, #8
  18:   e8bd 4080       ldmia.w sp!, {r7, lr}
  1c:   4671            mov     r1, lr
  1e:   4672            mov     r2, lr
  20:   4673            mov     r3, lr
  22:   46f4            mov     ip, lr
  24:   f38e 8800       msr     CPSR_f, lr
  28:   4774            bxns    lr
  2a:   f240 0000       movw    r0, #0
  2e:   f2c0 0000       movt    r0, #0
  32:   f240 0200       movw    r2, #0
  36:   f2c0 0200       movt    r2, #0
  3a:   211c            movs    r1, #28
  3c:   f7ff fffe       bl      0 <_ZN4core9panicking5panic17h5c028258ca2fb3f5E>
  40:   defe            udf     #254    ; 0xfe
```
"##,
    },
    Lint {
        label: "compiler_builtins",
        description: r##"# `compiler_builtins`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "concat_idents",
        description: r##"# `concat_idents`

The tracking issue for this feature is: [#29599]

[#29599]: https://github.com/rust-lang/rust/issues/29599

------------------------

The `concat_idents` feature adds a macro for concatenating multiple identifiers
into one identifier.

## Examples

```rust
#![feature(concat_idents)]

fn main() {
    fn foobar() -> u32 { 23 }
    let f = concat_idents!(foo, bar);
    assert_eq!(f(), 23);
}
```
"##,
    },
    Lint {
        label: "const_eval_limit",
        description: r##"# `const_eval_limit`

The tracking issue for this feature is: [#67217]

[#67217]: https://github.com/rust-lang/rust/issues/67217

The `const_eval_limit` allows someone to limit the evaluation steps the CTFE undertakes to evaluate a `const fn`.
"##,
    },
    Lint {
        label: "core_intrinsics",
        description: r##"# `core_intrinsics`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "core_panic",
        description: r##"# `core_panic`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "core_private_bignum",
        description: r##"# `core_private_bignum`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "core_private_diy_float",
        description: r##"# `core_private_diy_float`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "crate_visibility_modifier",
        description: r##"# `crate_visibility_modifier`

The tracking issue for this feature is: [#53120]

[#53120]: https://github.com/rust-lang/rust/issues/53120

-----

The `crate_visibility_modifier` feature allows the `crate` keyword to be used
as a visibility modifier synonymous to `pub(crate)`, indicating that a type
(function, _&c._) is to be visible to the entire enclosing crate, but not to
other crates.

```rust
#![feature(crate_visibility_modifier)]

crate struct Foo {
    bar: usize,
}
```
"##,
    },
    Lint {
        label: "custom_test_frameworks",
        description: r##"# `custom_test_frameworks`

The tracking issue for this feature is: [#50297]

[#50297]: https://github.com/rust-lang/rust/issues/50297

------------------------

The `custom_test_frameworks` feature allows the use of `#[test_case]` and `#![test_runner]`.
Any function, const, or static can be annotated with `#[test_case]` causing it to be aggregated (like `#[test]`)
and be passed to the test runner determined by the `#![test_runner]` crate attribute.

```rust
#![feature(custom_test_frameworks)]
#![test_runner(my_runner)]

fn my_runner(tests: &[&i32]) {
    for t in tests {
        if **t == 0 {
            println!("PASSED");
        } else {
            println!("FAILED");
        }
    }
}

#[test_case]
const WILL_PASS: i32 = 0;

#[test_case]
const WILL_FAIL: i32 = 4;
```
"##,
    },
    Lint {
        label: "dec2flt",
        description: r##"# `dec2flt`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "default_free_fn",
        description: r##"# `default_free_fn`

The tracking issue for this feature is: [#73014]

[#73014]: https://github.com/rust-lang/rust/issues/73014

------------------------

Adds a free `default()` function to the `std::default` module.  This function
just forwards to [`Default::default()`], but may remove repetition of the word
"default" from the call site.

[`Default::default()`]: https://doc.rust-lang.org/nightly/std/default/trait.Default.html#tymethod.default

Here is an example:

```rust
#![feature(default_free_fn)]
use std::default::default;

#[derive(Default)]
struct AppConfig {
    foo: FooConfig,
    bar: BarConfig,
}

#[derive(Default)]
struct FooConfig {
    foo: i32,
}

#[derive(Default)]
struct BarConfig {
    bar: f32,
    baz: u8,
}

fn main() {
    let options = AppConfig {
        foo: default(),
        bar: BarConfig {
            bar: 10.1,
            ..default()
        },
    };
}
```
"##,
    },
    Lint {
        label: "derive_clone_copy",
        description: r##"# `derive_clone_copy`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "derive_eq",
        description: r##"# `derive_eq`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "doc_cfg",
        description: r##"# `doc_cfg`

The tracking issue for this feature is: [#43781]

------

The `doc_cfg` feature allows an API be documented as only available in some specific platforms.
This attribute has two effects:

1. In the annotated item's documentation, there will be a message saying "This is supported on
    (platform) only".

2. The item's doc-tests will only run on the specific platform.

In addition to allowing the use of the `#[doc(cfg)]` attribute, this feature enables the use of a
special conditional compilation flag, `#[cfg(doc)]`, set whenever building documentation on your
crate.

This feature was introduced as part of PR [#43348] to allow the platform-specific parts of the
standard library be documented.

```rust
#![feature(doc_cfg)]

#[cfg(any(windows, doc))]
#[doc(cfg(windows))]
/// The application's icon in the notification area (a.k.a. system tray).
///
/// # Examples
///
/// ```no_run
/// extern crate my_awesome_ui_library;
/// use my_awesome_ui_library::current_app;
/// use my_awesome_ui_library::windows::notification;
///
/// let icon = current_app().get::<notification::Icon>();
/// icon.show();
/// icon.show_message("Hello");
/// ```
pub struct Icon {
    // ...
}
```

[#43781]: https://github.com/rust-lang/rust/issues/43781
[#43348]: https://github.com/rust-lang/rust/issues/43348
"##,
    },
    Lint {
        label: "doc_masked",
        description: r##"# `doc_masked`

The tracking issue for this feature is: [#44027]

-----

The `doc_masked` feature allows a crate to exclude types from a given crate from appearing in lists
of trait implementations. The specifics of the feature are as follows:

1. When rustdoc encounters an `extern crate` statement annotated with a `#[doc(masked)]` attribute,
   it marks the crate as being masked.

2. When listing traits a given type implements, rustdoc ensures that traits from masked crates are
   not emitted into the documentation.

3. When listing types that implement a given trait, rustdoc ensures that types from masked crates
   are not emitted into the documentation.

This feature was introduced in PR [#44026] to ensure that compiler-internal and
implementation-specific types and traits were not included in the standard library's documentation.
Such types would introduce broken links into the documentation.

[#44026]: https://github.com/rust-lang/rust/pull/44026
[#44027]: https://github.com/rust-lang/rust/pull/44027
"##,
    },
    Lint {
        label: "doc_notable_trait",
        description: r##"# `doc_notable_trait`

The tracking issue for this feature is: [#45040]

The `doc_notable_trait` feature allows the use of the `#[doc(notable_trait)]`
attribute, which will display the trait in a "Notable traits" dialog for
functions returning types that implement the trait. For example, this attribute
is applied to the `Iterator`, `Future`, `io::Read`, and `io::Write` traits in
the standard library.

You can do this on your own traits like so:

```
#![feature(doc_notable_trait)]

#[doc(notable_trait)]
pub trait MyTrait {}

pub struct MyStruct;
impl MyTrait for MyStruct {}

/// The docs for this function will have a button that displays a dialog about
/// `MyStruct` implementing `MyTrait`.
pub fn my_fn() -> MyStruct { MyStruct }
```

This feature was originally implemented in PR [#45039].

See also its documentation in [the rustdoc book][rustdoc-book-notable_trait].

[#45040]: https://github.com/rust-lang/rust/issues/45040
[#45039]: https://github.com/rust-lang/rust/pull/45039
[rustdoc-book-notable_trait]: ../../rustdoc/unstable-features.html#adding-your-trait-to-the-notable-traits-dialog
"##,
    },
    Lint {
        label: "fd",
        description: r##"# `fd`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "fd_read",
        description: r##"# `fd_read`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "ffi_const",
        description: r##"# `ffi_const`

The tracking issue for this feature is: [#58328]

------

The `#[ffi_const]` attribute applies clang's `const` attribute to foreign
functions declarations.

That is, `#[ffi_const]` functions shall have no effects except for its return
value, which can only depend on the values of the function parameters, and is
not affected by changes to the observable state of the program.

Applying the `#[ffi_const]` attribute to a function that violates these
requirements is undefined behaviour.

This attribute enables Rust to perform common optimizations, like sub-expression
elimination, and it can avoid emitting some calls in repeated invocations of the
function with the same argument values regardless of other operations being
performed in between these functions calls (as opposed to `#[ffi_pure]`
functions).

## Pitfalls

A `#[ffi_const]` function can only read global memory that would not affect
its return value for the whole execution of the program (e.g. immutable global
memory). `#[ffi_const]` functions are referentially-transparent and therefore
more strict than `#[ffi_pure]` functions.

A common pitfall involves applying the `#[ffi_const]` attribute to a
function that reads memory through pointer arguments which do not necessarily
point to immutable global memory.

A `#[ffi_const]` function that returns unit has no effect on the abstract
machine's state, and a `#[ffi_const]` function cannot be `#[ffi_pure]`.

A `#[ffi_const]` function must not diverge, neither via a side effect (e.g. a
call to `abort`) nor by infinite loops.

When translating C headers to Rust FFI, it is worth verifying for which targets
the `const` attribute is enabled in those headers, and using the appropriate
`cfg` macros in the Rust side to match those definitions. While the semantics of
`const` are implemented identically by many C and C++ compilers, e.g., clang,
[GCC], [ARM C/C++ compiler], [IBM ILE C/C++], etc. they are not necessarily
implemented in this way on all of them. It is therefore also worth verifying
that the semantics of the C toolchain used to compile the binary being linked
against are compatible with those of the `#[ffi_const]`.

[#58328]: https://github.com/rust-lang/rust/issues/58328
[ARM C/C++ compiler]: http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0491c/Cacgigch.html
[GCC]: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-const-function-attribute
[IBM ILE C/C++]: https://www.ibm.com/support/knowledgecenter/fr/ssw_ibm_i_71/rzarg/fn_attrib_const.htm
"##,
    },
    Lint {
        label: "ffi_pure",
        description: r##"# `ffi_pure`

The tracking issue for this feature is: [#58329]

------

The `#[ffi_pure]` attribute applies clang's `pure` attribute to foreign
functions declarations.

That is, `#[ffi_pure]` functions shall have no effects except for its return
value, which shall not change across two consecutive function calls with
the same parameters.

Applying the `#[ffi_pure]` attribute to a function that violates these
requirements is undefined behavior.

This attribute enables Rust to perform common optimizations, like sub-expression
elimination and loop optimizations. Some common examples of pure functions are
`strlen` or `memcmp`.

These optimizations are only applicable when the compiler can prove that no
program state observable by the `#[ffi_pure]` function has changed between calls
of the function, which could alter the result. See also the `#[ffi_const]`
attribute, which provides stronger guarantees regarding the allowable behavior
of a function, enabling further optimization.

## Pitfalls

A `#[ffi_pure]` function can read global memory through the function
parameters (e.g. pointers), globals, etc. `#[ffi_pure]` functions are not
referentially-transparent, and are therefore more relaxed than `#[ffi_const]`
functions.

However, accessing global memory through volatile or atomic reads can violate the
requirement that two consecutive function calls shall return the same value.

A `pure` function that returns unit has no effect on the abstract machine's
state.

A `#[ffi_pure]` function must not diverge, neither via a side effect (e.g. a
call to `abort`) nor by infinite loops.

When translating C headers to Rust FFI, it is worth verifying for which targets
the `pure` attribute is enabled in those headers, and using the appropriate
`cfg` macros in the Rust side to match those definitions. While the semantics of
`pure` are implemented identically by many C and C++ compilers, e.g., clang,
[GCC], [ARM C/C++ compiler], [IBM ILE C/C++], etc. they are not necessarily
implemented in this way on all of them. It is therefore also worth verifying
that the semantics of the C toolchain used to compile the binary being linked
against are compatible with those of the `#[ffi_pure]`.


[#58329]: https://github.com/rust-lang/rust/issues/58329
[ARM C/C++ compiler]: http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0491c/Cacigdac.html
[GCC]: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-pure-function-attribute
[IBM ILE C/C++]: https://www.ibm.com/support/knowledgecenter/fr/ssw_ibm_i_71/rzarg/fn_attrib_pure.htm
"##,
    },
    Lint {
        label: "flt2dec",
        description: r##"# `flt2dec`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "fmt_internals",
        description: r##"# `fmt_internals`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "fn_traits",
        description: r##"# `fn_traits`

The tracking issue for this feature is [#29625]

See Also: [`unboxed_closures`](../language-features/unboxed-closures.md)

[#29625]: https://github.com/rust-lang/rust/issues/29625

----

The `fn_traits` feature allows for implementation of the [`Fn*`] traits
for creating custom closure-like types.

[`Fn*`]: https://doc.rust-lang.org/std/ops/trait.Fn.html

```rust
#![feature(unboxed_closures)]
#![feature(fn_traits)]

struct Adder {
    a: u32
}

impl FnOnce<(u32, )> for Adder {
    type Output = u32;
    extern "rust-call" fn call_once(self, b: (u32, )) -> Self::Output {
        self.a + b.0
    }
}

fn main() {
    let adder = Adder { a: 3 };
    assert_eq!(adder(2), 5);
}
```
"##,
    },
    Lint {
        label: "format_args_capture",
        description: r##"# `format_args_capture`

The tracking issue for this feature is: [#67984]

[#67984]: https://github.com/rust-lang/rust/issues/67984

------------------------

Enables `format_args!` (and macros which use `format_args!` in their implementation, such
as `format!`, `print!` and `panic!`) to capture variables from the surrounding scope.
This avoids the need to pass named parameters when the binding in question
already exists in scope.

```rust
#![feature(format_args_capture)]

let (person, species, name) = ("Charlie Brown", "dog", "Snoopy");

// captures named argument `person`
print!("Hello {person}");

// captures named arguments `species` and `name`
format!("The {species}'s name is {name}.");
```

This also works for formatting parameters such as width and precision:

```rust
#![feature(format_args_capture)]

let precision = 2;
let s = format!("{:.precision$}", 1.324223);

assert_eq!(&s, "1.32");
```

A non-exhaustive list of macros which benefit from this functionality include:
- `format!`
- `print!` and `println!`
- `eprint!` and `eprintln!`
- `write!` and `writeln!`
- `panic!`
- `unreachable!`
- `unimplemented!`
- `todo!`
- `assert!` and similar
- macros in many thirdparty crates, such as `log`
"##,
    },
    Lint {
        label: "generators",
        description: r##"# `generators`

The tracking issue for this feature is: [#43122]

[#43122]: https://github.com/rust-lang/rust/issues/43122

------------------------

The `generators` feature gate in Rust allows you to define generator or
coroutine literals. A generator is a "resumable function" that syntactically
resembles a closure but compiles to much different semantics in the compiler
itself. The primary feature of a generator is that it can be suspended during
execution to be resumed at a later date. Generators use the `yield` keyword to
"return", and then the caller can `resume` a generator to resume execution just
after the `yield` keyword.

Generators are an extra-unstable feature in the compiler right now. Added in
[RFC 2033] they're mostly intended right now as a information/constraint
gathering phase. The intent is that experimentation can happen on the nightly
compiler before actual stabilization. A further RFC will be required to
stabilize generators/coroutines and will likely contain at least a few small
tweaks to the overall design.

[RFC 2033]: https://github.com/rust-lang/rfcs/pull/2033

A syntactical example of a generator is:

```rust
#![feature(generators, generator_trait)]

use std::ops::{Generator, GeneratorState};
use std::pin::Pin;

fn main() {
    let mut generator = || {
        yield 1;
        return "foo"
    };

    match Pin::new(&mut generator).resume(()) {
        GeneratorState::Yielded(1) => {}
        _ => panic!("unexpected value from resume"),
    }
    match Pin::new(&mut generator).resume(()) {
        GeneratorState::Complete("foo") => {}
        _ => panic!("unexpected value from resume"),
    }
}
```

Generators are closure-like literals which can contain a `yield` statement. The
`yield` statement takes an optional expression of a value to yield out of the
generator. All generator literals implement the `Generator` trait in the
`std::ops` module. The `Generator` trait has one main method, `resume`, which
resumes execution of the generator at the previous suspension point.

An example of the control flow of generators is that the following example
prints all numbers in order:

```rust
#![feature(generators, generator_trait)]

use std::ops::Generator;
use std::pin::Pin;

fn main() {
    let mut generator = || {
        println!("2");
        yield;
        println!("4");
    };

    println!("1");
    Pin::new(&mut generator).resume(());
    println!("3");
    Pin::new(&mut generator).resume(());
    println!("5");
}
```

At this time the main intended use case of generators is an implementation
primitive for async/await syntax, but generators will likely be extended to
ergonomic implementations of iterators and other primitives in the future.
Feedback on the design and usage is always appreciated!

### The `Generator` trait

The `Generator` trait in `std::ops` currently looks like:

```rust
# #![feature(arbitrary_self_types, generator_trait)]
# use std::ops::GeneratorState;
# use std::pin::Pin;

pub trait Generator<R = ()> {
    type Yield;
    type Return;
    fn resume(self: Pin<&mut Self>, resume: R) -> GeneratorState<Self::Yield, Self::Return>;
}
```

The `Generator::Yield` type is the type of values that can be yielded with the
`yield` statement. The `Generator::Return` type is the returned type of the
generator. This is typically the last expression in a generator's definition or
any value passed to `return` in a generator. The `resume` function is the entry
point for executing the `Generator` itself.

The return value of `resume`, `GeneratorState`, looks like:

```rust
pub enum GeneratorState<Y, R> {
    Yielded(Y),
    Complete(R),
}
```

The `Yielded` variant indicates that the generator can later be resumed. This
corresponds to a `yield` point in a generator. The `Complete` variant indicates
that the generator is complete and cannot be resumed again. Calling `resume`
after a generator has returned `Complete` will likely result in a panic of the
program.

### Closure-like semantics

The closure-like syntax for generators alludes to the fact that they also have
closure-like semantics. Namely:

* When created, a generator executes no code. A closure literal does not
  actually execute any of the closure's code on construction, and similarly a
  generator literal does not execute any code inside the generator when
  constructed.

* Generators can capture outer variables by reference or by move, and this can
  be tweaked with the `move` keyword at the beginning of the closure. Like
  closures all generators will have an implicit environment which is inferred by
  the compiler. Outer variables can be moved into a generator for use as the
  generator progresses.

* Generator literals produce a value with a unique type which implements the
  `std::ops::Generator` trait. This allows actual execution of the generator
  through the `Generator::resume` method as well as also naming it in return
  types and such.

* Traits like `Send` and `Sync` are automatically implemented for a `Generator`
  depending on the captured variables of the environment. Unlike closures,
  generators also depend on variables live across suspension points. This means
  that although the ambient environment may be `Send` or `Sync`, the generator
  itself may not be due to internal variables live across `yield` points being
  not-`Send` or not-`Sync`. Note that generators do
  not implement traits like `Copy` or `Clone` automatically.

* Whenever a generator is dropped it will drop all captured environment
  variables.

### Generators as state machines

In the compiler, generators are currently compiled as state machines. Each
`yield` expression will correspond to a different state that stores all live
variables over that suspension point. Resumption of a generator will dispatch on
the current state and then execute internally until a `yield` is reached, at
which point all state is saved off in the generator and a value is returned.

Let's take a look at an example to see what's going on here:

```rust
#![feature(generators, generator_trait)]

use std::ops::Generator;
use std::pin::Pin;

fn main() {
    let ret = "foo";
    let mut generator = move || {
        yield 1;
        return ret
    };

    Pin::new(&mut generator).resume(());
    Pin::new(&mut generator).resume(());
}
```

This generator literal will compile down to something similar to:

```rust
#![feature(arbitrary_self_types, generators, generator_trait)]

use std::ops::{Generator, GeneratorState};
use std::pin::Pin;

fn main() {
    let ret = "foo";
    let mut generator = {
        enum __Generator {
            Start(&'static str),
            Yield1(&'static str),
            Done,
        }

        impl Generator for __Generator {
            type Yield = i32;
            type Return = &'static str;

            fn resume(mut self: Pin<&mut Self>, resume: ()) -> GeneratorState<i32, &'static str> {
                use std::mem;
                match mem::replace(&mut *self, __Generator::Done) {
                    __Generator::Start(s) => {
                        *self = __Generator::Yield1(s);
                        GeneratorState::Yielded(1)
                    }

                    __Generator::Yield1(s) => {
                        *self = __Generator::Done;
                        GeneratorState::Complete(s)
                    }

                    __Generator::Done => {
                        panic!("generator resumed after completion")
                    }
                }
            }
        }

        __Generator::Start(ret)
    };

    Pin::new(&mut generator).resume(());
    Pin::new(&mut generator).resume(());
}
```

Notably here we can see that the compiler is generating a fresh type,
`__Generator` in this case. This type has a number of states (represented here
as an `enum`) corresponding to each of the conceptual states of the generator.
At the beginning we're closing over our outer variable `foo` and then that
variable is also live over the `yield` point, so it's stored in both states.

When the generator starts it'll immediately yield 1, but it saves off its state
just before it does so indicating that it has reached the yield point. Upon
resuming again we'll execute the `return ret` which returns the `Complete`
state.

Here we can also note that the `Done` state, if resumed, panics immediately as
it's invalid to resume a completed generator. It's also worth noting that this
is just a rough desugaring, not a normative specification for what the compiler
does.
"##,
    },
    Lint {
        label: "global_asm",
        description: r##"# `global_asm`

The tracking issue for this feature is: [#35119]

[#35119]: https://github.com/rust-lang/rust/issues/35119

------------------------

The `global_asm!` macro allows the programmer to write arbitrary
assembly outside the scope of a function body, passing it through
`rustc` and `llvm` to the assembler. That is to say, `global_asm!` is
equivalent to assembling the asm with an external assembler and then
linking the resulting object file with the current crate.

`global_asm!` fills a role not currently satisfied by either `asm!`
or `#[naked]` functions. The programmer has _all_ features of the
assembler at their disposal. The linker will expect to resolve any
symbols defined in the inline assembly, modulo any symbols marked as
external. It also means syntax for directives and assembly follow the
conventions of the assembler in your toolchain.

A simple usage looks like this:

```rust,ignore (requires-external-file)
#![feature(global_asm)]
# // you also need relevant target_arch cfgs
global_asm!(include_str!("something_neato.s"));
```

And a more complicated usage looks like this:

```rust,no_run
#![feature(global_asm)]
# #[cfg(any(target_arch="x86", target_arch="x86_64"))]
# mod x86 {

pub mod sally {
    global_asm!(
        ".global foo",
        "foo:",
        "jmp baz",
    );

    #[no_mangle]
    pub unsafe extern "C" fn baz() {}
}

// the symbols `foo` and `bar` are global, no matter where
// `global_asm!` was used.
extern "C" {
    fn foo();
    fn bar();
}

pub mod harry {
    global_asm!(
        ".global bar",
        "bar:",
        "jmp quux",
    );

    #[no_mangle]
    pub unsafe extern "C" fn quux() {}
}
# }
```

You may use `global_asm!` multiple times, anywhere in your crate, in
whatever way suits you. However, you should not rely on assembler state
(e.g. assembler macros) defined in one `global_asm!` to be available in
another one. It is implementation-defined whether the multiple usages
are concatenated into one or assembled separately.

`global_asm!` also supports `const` operands like `asm!`, which allows
constants defined in Rust to be used in assembly code:

```rust,no_run
#![feature(global_asm)]
# #[cfg(any(target_arch="x86", target_arch="x86_64"))]
# mod x86 {
const C: i32 = 1234;
global_asm!(
    ".global bar",
    "bar: .word {c}",
    c = const C,
);
# }
```

The syntax for passing operands is the same as `asm!` except that only
`const` operands are allowed. Refer to the [asm](asm.md) documentation
for more details.

On x86, the assembly code will use intel syntax by default. You can
override this by adding `options(att_syntax)` at the end of the macro
arguments list:

```rust,no_run
#![feature(global_asm)]
# #[cfg(any(target_arch="x86", target_arch="x86_64"))]
# mod x86 {
global_asm!("movl ${}, %ecx", const 5, options(att_syntax));
// is equivalent to
global_asm!("mov ecx, {}", const 5);
# }
```

------------------------

If you don't need quite as much power and flexibility as
`global_asm!` provides, and you don't mind restricting your inline
assembly to `fn` bodies only, you might try the
[asm](asm.md) feature instead.
"##,
    },
    Lint {
        label: "impl_trait_in_bindings",
        description: r##"# `impl_trait_in_bindings`

The tracking issue for this feature is: [#63065]

[#63065]: https://github.com/rust-lang/rust/issues/63065

------------------------

The `impl_trait_in_bindings` feature gate lets you use `impl Trait` syntax in
`let`, `static`, and `const` bindings.

A simple example is:

```rust
#![feature(impl_trait_in_bindings)]

use std::fmt::Debug;

fn main() {
    let a: impl Debug + Clone = 42;
    let b = a.clone();
    println!("{:?}", b); // prints `42`
}
```

Note however that because the types of `a` and `b` are opaque in the above
example, calling inherent methods or methods outside of the specified traits
(e.g., `a.abs()` or `b.abs()`) is not allowed, and yields an error.
"##,
    },
    Lint {
        label: "infer_static_outlives_requirements",
        description: r##"# `infer_static_outlives_requirements`

The tracking issue for this feature is: [#54185]

[#54185]: https://github.com/rust-lang/rust/issues/54185

------------------------
The `infer_static_outlives_requirements` feature indicates that certain
`'static` outlives requirements can be inferred by the compiler rather than
stating them explicitly.

Note: It is an accompanying feature to `infer_outlives_requirements`,
which must be enabled to infer outlives requirements.

For example, currently generic struct definitions that contain
references, require where-clauses of the form T: 'static. By using
this feature the outlives predicates will be inferred, although
they may still be written explicitly.

```rust,ignore (pseudo-Rust)
struct Foo<U> where U: 'static { // <-- currently required
    bar: Bar<U>
}
struct Bar<T: 'static> {
    x: T,
}
```


## Examples:

```rust,ignore (pseudo-Rust)
#![feature(infer_outlives_requirements)]
#![feature(infer_static_outlives_requirements)]

#[rustc_outlives]
// Implicitly infer U: 'static
struct Foo<U> {
    bar: Bar<U>
}
struct Bar<T: 'static> {
    x: T,
}
```
"##,
    },
    Lint {
        label: "inline_const",
        description: r##"# `inline_const`

The tracking issue for this feature is: [#76001]

------

This feature allows you to use inline constant expressions. For example, you can
turn this code:

```rust
# fn add_one(x: i32) -> i32 { x + 1 }
const MY_COMPUTATION: i32 = 1 + 2 * 3 / 4;

fn main() {
    let x = add_one(MY_COMPUTATION);
}
```

into this code:

```rust
#![feature(inline_const)]

# fn add_one(x: i32) -> i32 { x + 1 }
fn main() {
    let x = add_one(const { 1 + 2 * 3 / 4 });
}
```

You can also use inline constant expressions in patterns:

```rust
#![feature(inline_const)]

const fn one() -> i32 { 1 }

let some_int = 3;
match some_int {
    const { 1 + 2 } => println!("Matched 1 + 2"),
    const { one() } => println!("Matched const fn returning 1"),
    _ => println!("Didn't match anything :("),
}
```

[#76001]: https://github.com/rust-lang/rust/issues/76001
"##,
    },
    Lint {
        label: "int_error_internals",
        description: r##"# `int_error_internals`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "internal_output_capture",
        description: r##"# `internal_output_capture`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "intra_doc_pointers",
        description: r##"# `intra-doc-pointers`

The tracking issue for this feature is: [#80896]

[#80896]: https://github.com/rust-lang/rust/issues/80896

------------------------

Rustdoc does not currently allow disambiguating between `*const` and `*mut`, and
raw pointers in intra-doc links are unstable until it does.

```rust
#![feature(intra_doc_pointers)]
//! [pointer::add]
```
"##,
    },
    Lint {
        label: "intrinsics",
        description: r##"# `intrinsics`

The tracking issue for this feature is: None.

Intrinsics are never intended to be stable directly, but intrinsics are often
exported in some sort of stable manner. Prefer using the stable interfaces to
the intrinsic directly when you can.

------------------------


These are imported as if they were FFI functions, with the special
`rust-intrinsic` ABI. For example, if one was in a freestanding
context, but wished to be able to `transmute` between types, and
perform efficient pointer arithmetic, one would import those functions
via a declaration like

```rust
#![feature(intrinsics)]
# fn main() {}

extern "rust-intrinsic" {
    fn transmute<T, U>(x: T) -> U;

    fn offset<T>(dst: *const T, offset: isize) -> *const T;
}
```

As with any other FFI functions, these are always `unsafe` to call.
"##,
    },
    Lint {
        label: "is_sorted",
        description: r##"# `is_sorted`

The tracking issue for this feature is: [#53485]

[#53485]: https://github.com/rust-lang/rust/issues/53485

------------------------

Add the methods `is_sorted`, `is_sorted_by` and `is_sorted_by_key` to `[T]`;
add the methods `is_sorted`, `is_sorted_by` and `is_sorted_by_key` to
`Iterator`.
"##,
    },
    Lint {
        label: "lang_items",
        description: r##"# `lang_items`

The tracking issue for this feature is: None.

------------------------

The `rustc` compiler has certain pluggable operations, that is,
functionality that isn't hard-coded into the language, but is
implemented in libraries, with a special marker to tell the compiler
it exists. The marker is the attribute `#[lang = "..."]` and there are
various different values of `...`, i.e. various different 'lang
items'.

For example, `Box` pointers require two lang items, one for allocation
and one for deallocation. A freestanding program that uses the `Box`
sugar for dynamic allocations via `malloc` and `free`:

```rust,ignore (libc-is-finicky)
#![feature(lang_items, box_syntax, start, libc, core_intrinsics, rustc_private)]
#![no_std]
use core::intrinsics;
use core::panic::PanicInfo;

extern crate libc;

#[lang = "owned_box"]
pub struct Box<T>(*mut T);

#[lang = "exchange_malloc"]
unsafe fn allocate(size: usize, _align: usize) -> *mut u8 {
    let p = libc::malloc(size as libc::size_t) as *mut u8;

    // Check if `malloc` failed:
    if p as usize == 0 {
        intrinsics::abort();
    }

    p
}

#[lang = "box_free"]
unsafe fn box_free<T: ?Sized>(ptr: *mut T) {
    libc::free(ptr as *mut libc::c_void)
}

#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize {
    let _x = box 1;

    0
}

#[lang = "eh_personality"] extern fn rust_eh_personality() {}
#[lang = "panic_impl"] extern fn rust_begin_panic(info: &PanicInfo) -> ! { unsafe { intrinsics::abort() } }
#[no_mangle] pub extern fn rust_eh_register_frames () {}
#[no_mangle] pub extern fn rust_eh_unregister_frames () {}
```

Note the use of `abort`: the `exchange_malloc` lang item is assumed to
return a valid pointer, and so needs to do the check internally.

Other features provided by lang items include:

- overloadable operators via traits: the traits corresponding to the
  `==`, `<`, dereferencing (`*`) and `+` (etc.) operators are all
  marked with lang items; those specific four are `eq`, `ord`,
  `deref`, and `add` respectively.
- stack unwinding and general failure; the `eh_personality`,
  `panic` and `panic_bounds_check` lang items.
- the traits in `std::marker` used to indicate types of
  various kinds; lang items `send`, `sync` and `copy`.
- the marker types and variance indicators found in
  `std::marker`; lang items `covariant_type`,
  `contravariant_lifetime`, etc.

Lang items are loaded lazily by the compiler; e.g. if one never uses
`Box` then there is no need to define functions for `exchange_malloc`
and `box_free`. `rustc` will emit an error when an item is needed
but not found in the current crate or any that it depends on.

Most lang items are defined by `libcore`, but if you're trying to build
an executable without the standard library, you'll run into the need
for lang items. The rest of this page focuses on this use-case, even though
lang items are a bit broader than that.

### Using libc

In order to build a `#[no_std]` executable we will need libc as a dependency.
We can specify this using our `Cargo.toml` file:

```toml
[dependencies]
libc = { version = "0.2.14", default-features = false }
```

Note that the default features have been disabled. This is a critical step -
**the default features of libc include the standard library and so must be
disabled.**

### Writing an executable without stdlib

Controlling the entry point is possible in two ways: the `#[start]` attribute,
or overriding the default shim for the C `main` function with your own.

The function marked `#[start]` is passed the command line parameters
in the same format as C:

```rust,ignore (libc-is-finicky)
#![feature(lang_items, core_intrinsics, rustc_private)]
#![feature(start)]
#![no_std]
use core::intrinsics;
use core::panic::PanicInfo;

// Pull in the system libc library for what crt0.o likely requires.
extern crate libc;

// Entry point for this program.
#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    0
}

// These functions are used by the compiler, but not
// for a bare-bones hello world. These are normally
// provided by libstd.
#[lang = "eh_personality"]
#[no_mangle]
pub extern fn rust_eh_personality() {
}

#[lang = "panic_impl"]
#[no_mangle]
pub extern fn rust_begin_panic(info: &PanicInfo) -> ! {
    unsafe { intrinsics::abort() }
}
```

To override the compiler-inserted `main` shim, one has to disable it
with `#![no_main]` and then create the appropriate symbol with the
correct ABI and the correct name, which requires overriding the
compiler's name mangling too:

```rust,ignore (libc-is-finicky)
#![feature(lang_items, core_intrinsics, rustc_private)]
#![feature(start)]
#![no_std]
#![no_main]
use core::intrinsics;
use core::panic::PanicInfo;

// Pull in the system libc library for what crt0.o likely requires.
extern crate libc;

// Entry point for this program.
#[no_mangle] // ensure that this symbol is called `main` in the output
pub extern fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    0
}

// These functions are used by the compiler, but not
// for a bare-bones hello world. These are normally
// provided by libstd.
#[lang = "eh_personality"]
#[no_mangle]
pub extern fn rust_eh_personality() {
}

#[lang = "panic_impl"]
#[no_mangle]
pub extern fn rust_begin_panic(info: &PanicInfo) -> ! {
    unsafe { intrinsics::abort() }
}
```

In many cases, you may need to manually link to the `compiler_builtins` crate
when building a `no_std` binary. You may observe this via linker error messages
such as "```undefined reference to `__rust_probestack'```".

## More about the language items

The compiler currently makes a few assumptions about symbols which are
available in the executable to call. Normally these functions are provided by
the standard library, but without it you must define your own. These symbols
are called "language items", and they each have an internal name, and then a
signature that an implementation must conform to.

The first of these functions, `rust_eh_personality`, is used by the failure
mechanisms of the compiler. This is often mapped to GCC's personality function
(see the [libstd implementation][unwind] for more information), but crates
which do not trigger a panic can be assured that this function is never
called. The language item's name is `eh_personality`.

[unwind]: https://github.com/rust-lang/rust/blob/master/library/panic_unwind/src/gcc.rs

The second function, `rust_begin_panic`, is also used by the failure mechanisms of the
compiler. When a panic happens, this controls the message that's displayed on
the screen. While the language item's name is `panic_impl`, the symbol name is
`rust_begin_panic`.

Finally, a `eh_catch_typeinfo` static is needed for certain targets which
implement Rust panics on top of C++ exceptions.

## List of all language items

This is a list of all language items in Rust along with where they are located in
the source code.

- Primitives
  - `i8`: `libcore/num/mod.rs`
  - `i16`: `libcore/num/mod.rs`
  - `i32`: `libcore/num/mod.rs`
  - `i64`: `libcore/num/mod.rs`
  - `i128`: `libcore/num/mod.rs`
  - `isize`: `libcore/num/mod.rs`
  - `u8`: `libcore/num/mod.rs`
  - `u16`: `libcore/num/mod.rs`
  - `u32`: `libcore/num/mod.rs`
  - `u64`: `libcore/num/mod.rs`
  - `u128`: `libcore/num/mod.rs`
  - `usize`: `libcore/num/mod.rs`
  - `f32`: `libstd/f32.rs`
  - `f64`: `libstd/f64.rs`
  - `char`: `libcore/char.rs`
  - `slice`: `liballoc/slice.rs`
  - `str`: `liballoc/str.rs`
  - `const_ptr`: `libcore/ptr.rs`
  - `mut_ptr`: `libcore/ptr.rs`
  - `unsafe_cell`: `libcore/cell.rs`
- Runtime
  - `start`: `libstd/rt.rs`
  - `eh_personality`: `libpanic_unwind/emcc.rs` (EMCC)
  - `eh_personality`: `libpanic_unwind/gcc.rs` (GNU)
  - `eh_personality`: `libpanic_unwind/seh.rs` (SEH)
  - `eh_catch_typeinfo`: `libpanic_unwind/emcc.rs` (EMCC)
  - `panic`: `libcore/panicking.rs`
  - `panic_bounds_check`: `libcore/panicking.rs`
  - `panic_impl`: `libcore/panicking.rs`
  - `panic_impl`: `libstd/panicking.rs`
- Allocations
  - `owned_box`: `liballoc/boxed.rs`
  - `exchange_malloc`: `liballoc/heap.rs`
  - `box_free`: `liballoc/heap.rs`
- Operands
  - `not`: `libcore/ops/bit.rs`
  - `bitand`: `libcore/ops/bit.rs`
  - `bitor`: `libcore/ops/bit.rs`
  - `bitxor`: `libcore/ops/bit.rs`
  - `shl`: `libcore/ops/bit.rs`
  - `shr`: `libcore/ops/bit.rs`
  - `bitand_assign`: `libcore/ops/bit.rs`
  - `bitor_assign`: `libcore/ops/bit.rs`
  - `bitxor_assign`: `libcore/ops/bit.rs`
  - `shl_assign`: `libcore/ops/bit.rs`
  - `shr_assign`: `libcore/ops/bit.rs`
  - `deref`: `libcore/ops/deref.rs`
  - `deref_mut`: `libcore/ops/deref.rs`
  - `index`: `libcore/ops/index.rs`
  - `index_mut`: `libcore/ops/index.rs`
  - `add`: `libcore/ops/arith.rs`
  - `sub`: `libcore/ops/arith.rs`
  - `mul`: `libcore/ops/arith.rs`
  - `div`: `libcore/ops/arith.rs`
  - `rem`: `libcore/ops/arith.rs`
  - `neg`: `libcore/ops/arith.rs`
  - `add_assign`: `libcore/ops/arith.rs`
  - `sub_assign`: `libcore/ops/arith.rs`
  - `mul_assign`: `libcore/ops/arith.rs`
  - `div_assign`: `libcore/ops/arith.rs`
  - `rem_assign`: `libcore/ops/arith.rs`
  - `eq`: `libcore/cmp.rs`
  - `ord`: `libcore/cmp.rs`
- Functions
  - `fn`: `libcore/ops/function.rs`
  - `fn_mut`: `libcore/ops/function.rs`
  - `fn_once`: `libcore/ops/function.rs`
  - `generator_state`: `libcore/ops/generator.rs`
  - `generator`: `libcore/ops/generator.rs`
- Other
  - `coerce_unsized`: `libcore/ops/unsize.rs`
  - `drop`: `libcore/ops/drop.rs`
  - `drop_in_place`: `libcore/ptr.rs`
  - `clone`: `libcore/clone.rs`
  - `copy`: `libcore/marker.rs`
  - `send`: `libcore/marker.rs`
  - `sized`: `libcore/marker.rs`
  - `unsize`: `libcore/marker.rs`
  - `sync`: `libcore/marker.rs`
  - `phantom_data`: `libcore/marker.rs`
  - `discriminant_kind`: `libcore/marker.rs`
  - `freeze`: `libcore/marker.rs`
  - `debug_trait`: `libcore/fmt/mod.rs`
  - `non_zero`: `libcore/nonzero.rs`
  - `arc`: `liballoc/sync.rs`
  - `rc`: `liballoc/rc.rs`
"##,
    },
    Lint {
        label: "libstd_sys_internals",
        description: r##"# `libstd_sys_internals`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "libstd_thread_internals",
        description: r##"# `libstd_thread_internals`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "link_cfg",
        description: r##"# `link_cfg`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "llvm_asm",
        description: r##"# `llvm_asm`

The tracking issue for this feature is: [#70173]

[#70173]: https://github.com/rust-lang/rust/issues/70173

------------------------

For extremely low-level manipulations and performance reasons, one
might wish to control the CPU directly. Rust supports using inline
assembly to do this via the `llvm_asm!` macro.

```rust,ignore (pseudo-code)
llvm_asm!(assembly template
   : output operands
   : input operands
   : clobbers
   : options
   );
```

Any use of `llvm_asm` is feature gated (requires `#![feature(llvm_asm)]` on the
crate to allow) and of course requires an `unsafe` block.

> **Note**: the examples here are given in x86/x86-64 assembly, but
> all platforms are supported.

## Assembly template

The `assembly template` is the only required parameter and must be a
literal string (i.e. `""`)

```rust
#![feature(llvm_asm)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn foo() {
    unsafe {
        llvm_asm!("NOP");
    }
}

// Other platforms:
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn foo() { /* ... */ }

fn main() {
    // ...
    foo();
    // ...
}
```

(The `feature(llvm_asm)` and `#[cfg]`s are omitted from now on.)

Output operands, input operands, clobbers and options are all optional
but you must add the right number of `:` if you skip them:

```rust
# #![feature(llvm_asm)]
# #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
# fn main() { unsafe {
llvm_asm!("xor %eax, %eax"
    :
    :
    : "eax"
   );
# } }
# #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
# fn main() {}
```

Whitespace also doesn't matter:

```rust
# #![feature(llvm_asm)]
# #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
# fn main() { unsafe {
llvm_asm!("xor %eax, %eax" ::: "eax");
# } }
# #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
# fn main() {}
```

## Operands

Input and output operands follow the same format: `:
"constraints1"(expr1), "constraints2"(expr2), ..."`. Output operand
expressions must be mutable place, or not yet assigned:

```rust
# #![feature(llvm_asm)]
# #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn add(a: i32, b: i32) -> i32 {
    let c: i32;
    unsafe {
        llvm_asm!("add $2, $0"
             : "=r"(c)
             : "0"(a), "r"(b)
             );
    }
    c
}
# #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
# fn add(a: i32, b: i32) -> i32 { a + b }

fn main() {
    assert_eq!(add(3, 14159), 14162)
}
```

If you would like to use real operands in this position, however,
you are required to put curly braces `{}` around the register that
you want, and you are required to put the specific size of the
operand. This is useful for very low level programming, where
which register you use is important:

```rust
# #![feature(llvm_asm)]
# #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
# unsafe fn read_byte_in(port: u16) -> u8 {
let result: u8;
llvm_asm!("in %dx, %al" : "={al}"(result) : "{dx}"(port));
result
# }
```

## Clobbers

Some instructions modify registers which might otherwise have held
different values so we use the clobbers list to indicate to the
compiler not to assume any values loaded into those registers will
stay valid.

```rust
# #![feature(llvm_asm)]
# #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
# fn main() { unsafe {
// Put the value 0x200 in eax:
llvm_asm!("mov $$0x200, %eax" : /* no outputs */ : /* no inputs */ : "eax");
# } }
# #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
# fn main() {}
```

Input and output registers need not be listed since that information
is already communicated by the given constraints. Otherwise, any other
registers used either implicitly or explicitly should be listed.

If the assembly changes the condition code register `cc` should be
specified as one of the clobbers. Similarly, if the assembly modifies
memory, `memory` should also be specified.

## Options

The last section, `options` is specific to Rust. The format is comma
separated literal strings (i.e. `:"foo", "bar", "baz"`). It's used to
specify some extra info about the inline assembly:

Current valid options are:

1. `volatile` - specifying this is analogous to
   `__asm__ __volatile__ (...)` in gcc/clang.
2. `alignstack` - certain instructions expect the stack to be
   aligned a certain way (i.e. SSE) and specifying this indicates to
   the compiler to insert its usual stack alignment code
3. `intel` - use intel syntax instead of the default AT&T.

```rust
# #![feature(llvm_asm)]
# #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
# fn main() {
let result: i32;
unsafe {
   llvm_asm!("mov eax, 2" : "={eax}"(result) : : : "intel")
}
println!("eax is currently {}", result);
# }
# #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
# fn main() {}
```

## More Information

The current implementation of the `llvm_asm!` macro is a direct binding to [LLVM's
inline assembler expressions][llvm-docs], so be sure to check out [their
documentation as well][llvm-docs] for more information about clobbers,
constraints, etc.

[llvm-docs]: http://llvm.org/docs/LangRef.html#inline-assembler-expressions

If you need more power and don't mind losing some of the niceties of
`llvm_asm!`, check out [global_asm](global-asm.md).
"##,
    },
    Lint {
        label: "marker_trait_attr",
        description: r##"# `marker_trait_attr`

The tracking issue for this feature is: [#29864]

[#29864]: https://github.com/rust-lang/rust/issues/29864

------------------------

Normally, Rust keeps you from adding trait implementations that could
overlap with each other, as it would be ambiguous which to use.  This
feature, however, carves out an exception to that rule: a trait can
opt-in to having overlapping implementations, at the cost that those
implementations are not allowed to override anything (and thus the
trait itself cannot have any associated items, as they're pointless
when they'd need to do the same thing for every type anyway).

```rust
#![feature(marker_trait_attr)]

#[marker] trait CheapToClone: Clone {}

impl<T: Copy> CheapToClone for T {}

// These could potentially overlap with the blanket implementation above,
// so are only allowed because CheapToClone is a marker trait.
impl<T: CheapToClone, U: CheapToClone> CheapToClone for (T, U) {}
impl<T: CheapToClone> CheapToClone for std::ops::Range<T> {}

fn cheap_clone<T: CheapToClone>(t: T) -> T {
    t.clone()
}
```

This is expected to replace the unstable `overlapping_marker_traits`
feature, which applied to all empty traits (without needing an opt-in).
"##,
    },
    Lint {
        label: "native_link_modifiers",
        description: r##"# `native_link_modifiers`

The tracking issue for this feature is: [#81490]

[#81490]: https://github.com/rust-lang/rust/issues/81490

------------------------

The `native_link_modifiers` feature allows you to use the `modifiers` syntax with the `#[link(..)]` attribute.

Modifiers are specified as a comma-delimited string with each modifier prefixed with either a `+` or `-` to indicate that the modifier is enabled or disabled, respectively. The last boolean value specified for a given modifier wins.
"##,
    },
    Lint {
        label: "native_link_modifiers_as_needed",
        description: r##"# `native_link_modifiers_as_needed`

The tracking issue for this feature is: [#81490]

[#81490]: https://github.com/rust-lang/rust/issues/81490

------------------------

The `native_link_modifiers_as_needed` feature allows you to use the `as-needed` modifier.

`as-needed` is only compatible with the `dynamic` and `framework` linking kinds. Using any other kind will result in a compiler error.

`+as-needed` means that the library will be actually linked only if it satisfies some undefined symbols at the point at which it is specified on the command line, making it similar to static libraries in this regard.

This modifier translates to `--as-needed` for ld-like linkers, and to `-dead_strip_dylibs` / `-needed_library` / `-needed_framework` for ld64.
The modifier does nothing for linkers that don't support it (e.g. `link.exe`).

The default for this modifier is unclear, some targets currently specify it as `+as-needed`, some do not. We may want to try making `+as-needed` a default for all targets.
"##,
    },
    Lint {
        label: "native_link_modifiers_bundle",
        description: r##"# `native_link_modifiers_bundle`

The tracking issue for this feature is: [#81490]

[#81490]: https://github.com/rust-lang/rust/issues/81490

------------------------

The `native_link_modifiers_bundle` feature allows you to use the `bundle` modifier.

Only compatible with the `static` linking kind. Using any other kind will result in a compiler error.

`+bundle` means objects from the static library are bundled into the produced crate (a rlib, for example) and are used from this crate later during linking of the final binary.

`-bundle` means the static library is included into the produced rlib "by name" and object files from it are included only during linking of the final binary, the file search by that name is also performed during final linking.

This modifier is supposed to supersede the `static-nobundle` linking kind defined by [RFC 1717](https://github.com/rust-lang/rfcs/pull/1717).

The default for this modifier is currently `+bundle`, but it could be changed later on some future edition boundary.
"##,
    },
    Lint {
        label: "native_link_modifiers_verbatim",
        description: r##"# `native_link_modifiers_verbatim`

The tracking issue for this feature is: [#81490]

[#81490]: https://github.com/rust-lang/rust/issues/81490

------------------------

The `native_link_modifiers_verbatim` feature allows you to use the `verbatim` modifier.

`+verbatim` means that rustc itself won't add any target-specified library prefixes or suffixes (like `lib` or `.a`) to the library name, and will try its best to ask for the same thing from the linker.

For `ld`-like linkers rustc will use the `-l:filename` syntax (note the colon) when passing the library, so the linker won't add any prefixes or suffixes as well.
See [`-l namespec`](https://sourceware.org/binutils/docs/ld/Options.html) in ld documentation for more details.
For linkers not supporting any verbatim modifiers (e.g. `link.exe` or `ld64`) the library name will be passed as is.

The default for this modifier is `-verbatim`.

This RFC changes the behavior of `raw-dylib` linking kind specified by [RFC 2627](https://github.com/rust-lang/rfcs/pull/2627). The `.dll` suffix (or other target-specified suffixes for other targets) is now added automatically.
If your DLL doesn't have the `.dll` suffix, it can be specified with `+verbatim`.
"##,
    },
    Lint {
        label: "native_link_modifiers_whole_archive",
        description: r##"# `native_link_modifiers_whole_archive`

The tracking issue for this feature is: [#81490]

[#81490]: https://github.com/rust-lang/rust/issues/81490

------------------------

The `native_link_modifiers_whole_archive` feature allows you to use the `whole-archive` modifier.

Only compatible with the `static` linking kind. Using any other kind will result in a compiler error.

`+whole-archive` means that the static library is linked as a whole archive without throwing any object files away.

This modifier translates to `--whole-archive` for `ld`-like linkers, to `/WHOLEARCHIVE` for `link.exe`, and to `-force_load` for `ld64`.
The modifier does nothing for linkers that don't support it.

The default for this modifier is `-whole-archive`.
"##,
    },
    Lint {
        label: "negative_impls",
        description: r##"# `negative_impls`

The tracking issue for this feature is [#68318].

[#68318]: https://github.com/rust-lang/rust/issues/68318

----

With the feature gate `negative_impls`, you can write negative impls as well as positive ones:

```rust
#![feature(negative_impls)]
trait DerefMut { }
impl<T: ?Sized> !DerefMut for &T { }
```

Negative impls indicate a semver guarantee that the given trait will not be implemented for the given types. Negative impls play an additional purpose for auto traits, described below.

Negative impls have the following characteristics:

* They do not have any items.
* They must obey the orphan rules as if they were a positive impl.
* They cannot "overlap" with any positive impls.

## Semver interaction

It is a breaking change to remove a negative impl. Negative impls are a commitment not to implement the given trait for the named types.

## Orphan and overlap rules

Negative impls must obey the same orphan rules as a positive impl. This implies you cannot add a negative impl for types defined in upstream crates and so forth.

Similarly, negative impls cannot overlap with positive impls, again using the same "overlap" check that we ordinarily use to determine if two impls overlap. (Note that positive impls typically cannot overlap with one another either, except as permitted by specialization.)

## Interaction with auto traits

Declaring a negative impl `impl !SomeAutoTrait for SomeType` for an
auto-trait serves two purposes:

* as with any trait, it declares that `SomeType` will never implement `SomeAutoTrait`;
* it disables the automatic `SomeType: SomeAutoTrait` impl that would otherwise have been generated.

Note that, at present, there is no way to indicate that a given type
does not implement an auto trait *but that it may do so in the
future*. For ordinary types, this is done by simply not declaring any
impl at all, but that is not an option for auto traits. A workaround
is that one could embed a marker type as one of the fields, where the
marker type is `!AutoTrait`.

## Immediate uses

Negative impls are used to declare that `&T: !DerefMut`  and `&mut T: !Clone`, as required to fix the soundness of `Pin` described in [#66544](https://github.com/rust-lang/rust/issues/66544).

This serves two purposes:

* For proving the correctness of unsafe code, we can use that impl as evidence that no `DerefMut` or `Clone` impl exists.
* It prevents downstream crates from creating such impls.
"##,
    },
    Lint {
        label: "no_coverage",
        description: r##"# `no_coverage`

The tracking issue for this feature is: [#84605]

[#84605]: https://github.com/rust-lang/rust/issues/84605

---

The `no_coverage` attribute can be used to selectively disable coverage
instrumentation in an annotated function. This might be useful to:

-   Avoid instrumentation overhead in a performance critical function
-   Avoid generating coverage for a function that is not meant to be executed,
    but still target 100% coverage for the rest of the program.

## Example

```rust
#![feature(no_coverage)]

// `foo()` will get coverage instrumentation (by default)
fn foo() {
  // ...
}

#[no_coverage]
fn bar() {
  // ...
}
```
"##,
    },
    Lint {
        label: "no_sanitize",
        description: r##"# `no_sanitize`

The tracking issue for this feature is: [#39699]

[#39699]: https://github.com/rust-lang/rust/issues/39699

------------------------

The `no_sanitize` attribute can be used to selectively disable sanitizer
instrumentation in an annotated function. This might be useful to: avoid
instrumentation overhead in a performance critical function, or avoid
instrumenting code that contains constructs unsupported by given sanitizer.

The precise effect of this annotation depends on particular sanitizer in use.
For example, with `no_sanitize(thread)`, the thread sanitizer will no longer
instrument non-atomic store / load operations, but it will instrument atomic
operations to avoid reporting false positives and provide meaning full stack
traces.

## Examples

``` rust
#![feature(no_sanitize)]

#[no_sanitize(address)]
fn foo() {
  // ...
}
```
"##,
    },
    Lint {
        label: "plugin",
        description: r##"# `plugin`

The tracking issue for this feature is: [#29597]

[#29597]: https://github.com/rust-lang/rust/issues/29597


This feature is part of "compiler plugins." It will often be used with the
[`plugin_registrar`] and `rustc_private` features.

[`plugin_registrar`]: plugin-registrar.md

------------------------

`rustc` can load compiler plugins, which are user-provided libraries that
extend the compiler's behavior with new lint checks, etc.

A plugin is a dynamic library crate with a designated *registrar* function that
registers extensions with `rustc`. Other crates can load these extensions using
the crate attribute `#![plugin(...)]`.  See the
`rustc_driver::plugin` documentation for more about the
mechanics of defining and loading a plugin.

In the vast majority of cases, a plugin should *only* be used through
`#![plugin]` and not through an `extern crate` item.  Linking a plugin would
pull in all of librustc_ast and librustc as dependencies of your crate.  This is
generally unwanted unless you are building another plugin.

The usual practice is to put compiler plugins in their own crate, separate from
any `macro_rules!` macros or ordinary Rust code meant to be used by consumers
of a library.

# Lint plugins

Plugins can extend [Rust's lint
infrastructure](../../reference/attributes/diagnostics.md#lint-check-attributes) with
additional checks for code style, safety, etc. Now let's write a plugin
[`lint-plugin-test.rs`](https://github.com/rust-lang/rust/blob/master/src/test/ui-fulldeps/auxiliary/lint-plugin-test.rs)
that warns about any item named `lintme`.

```rust,ignore (requires-stage-2)
#![feature(plugin_registrar)]
#![feature(box_syntax, rustc_private)]

extern crate rustc_ast;

// Load rustc as a plugin to get macros
extern crate rustc_driver;
#[macro_use]
extern crate rustc_lint;
#[macro_use]
extern crate rustc_session;

use rustc_driver::plugin::Registry;
use rustc_lint::{EarlyContext, EarlyLintPass, LintArray, LintContext, LintPass};
use rustc_ast::ast;
declare_lint!(TEST_LINT, Warn, "Warn about items named 'lintme'");

declare_lint_pass!(Pass => [TEST_LINT]);

impl EarlyLintPass for Pass {
    fn check_item(&mut self, cx: &EarlyContext, it: &ast::Item) {
        if it.ident.name.as_str() == "lintme" {
            cx.lint(TEST_LINT, |lint| {
                lint.build("item is named 'lintme'").set_span(it.span).emit()
            });
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.lint_store.register_lints(&[&TEST_LINT]);
    reg.lint_store.register_early_pass(|| box Pass);
}
```

Then code like

```rust,ignore (requires-plugin)
#![feature(plugin)]
#![plugin(lint_plugin_test)]

fn lintme() { }
```

will produce a compiler warning:

```txt
foo.rs:4:1: 4:16 warning: item is named 'lintme', #[warn(test_lint)] on by default
foo.rs:4 fn lintme() { }
         ^~~~~~~~~~~~~~~
```

The components of a lint plugin are:

* one or more `declare_lint!` invocations, which define static `Lint` structs;

* a struct holding any state needed by the lint pass (here, none);

* a `LintPass`
  implementation defining how to check each syntax element. A single
  `LintPass` may call `span_lint` for several different `Lint`s, but should
  register them all through the `get_lints` method.

Lint passes are syntax traversals, but they run at a late stage of compilation
where type information is available. `rustc`'s [built-in
lints](https://github.com/rust-lang/rust/blob/master/src/librustc_session/lint/builtin.rs)
mostly use the same infrastructure as lint plugins, and provide examples of how
to access type information.

Lints defined by plugins are controlled by the usual [attributes and compiler
flags](../../reference/attributes/diagnostics.md#lint-check-attributes), e.g.
`#[allow(test_lint)]` or `-A test-lint`. These identifiers are derived from the
first argument to `declare_lint!`, with appropriate case and punctuation
conversion.

You can run `rustc -W help foo.rs` to see a list of lints known to `rustc`,
including those provided by plugins loaded by `foo.rs`.
"##,
    },
    Lint {
        label: "plugin_registrar",
        description: r##"# `plugin_registrar`

The tracking issue for this feature is: [#29597]

[#29597]: https://github.com/rust-lang/rust/issues/29597

This feature is part of "compiler plugins." It will often be used with the
[`plugin`] and `rustc_private` features as well. For more details, see
their docs.

[`plugin`]: plugin.md

------------------------
"##,
    },
    Lint {
        label: "print_internals",
        description: r##"# `print_internals`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "profiler_runtime",
        description: r##"# `profiler_runtime`

The tracking issue for this feature is: [#42524](https://github.com/rust-lang/rust/issues/42524).

------------------------
"##,
    },
    Lint {
        label: "profiler_runtime_lib",
        description: r##"# `profiler_runtime_lib`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "repr128",
        description: r##"# `repr128`

The tracking issue for this feature is: [#56071]

[#56071]: https://github.com/rust-lang/rust/issues/56071

------------------------

The `repr128` feature adds support for `#[repr(u128)]` on `enum`s.

```rust
#![feature(repr128)]

#[repr(u128)]
enum Foo {
    Bar(u64),
}
```
"##,
    },
    Lint {
        label: "rt",
        description: r##"# `rt`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "rustc_attrs",
        description: r##"# `rustc_attrs`

This feature has no tracking issue, and is therefore internal to
the compiler, not being intended for general use.

Note: `rustc_attrs` enables many rustc-internal attributes and this page
only discuss a few of them.

------------------------

The `rustc_attrs` feature allows debugging rustc type layouts by using
`#[rustc_layout(...)]` to debug layout at compile time (it even works
with `cargo check`) as an alternative to `rustc -Z print-type-sizes`
that is way more verbose.

Options provided by `#[rustc_layout(...)]` are `debug`, `size`, `align`,
`abi`. Note that it only works on sized types without generics.

## Examples

```rust,compile_fail
#![feature(rustc_attrs)]

#[rustc_layout(abi, size)]
pub enum X {
    Y(u8, u8, u8),
    Z(isize),
}
```

When that is compiled, the compiler will error with something like

```text
error: abi: Aggregate { sized: true }
 --> src/lib.rs:4:1
  |
4 | / pub enum T {
5 | |     Y(u8, u8, u8),
6 | |     Z(isize),
7 | | }
  | |_^

error: size: Size { raw: 16 }
 --> src/lib.rs:4:1
  |
4 | / pub enum T {
5 | |     Y(u8, u8, u8),
6 | |     Z(isize),
7 | | }
  | |_^

error: aborting due to 2 previous errors
```
"##,
    },
    Lint {
        label: "sort_internals",
        description: r##"# `sort_internals`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "str_internals",
        description: r##"# `str_internals`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "test",
        description: r##"# `test`

The tracking issue for this feature is: None.

------------------------

The internals of the `test` crate are unstable, behind the `test` flag.  The
most widely used part of the `test` crate are benchmark tests, which can test
the performance of your code.  Let's make our `src/lib.rs` look like this
(comments elided):

```rust,no_run
#![feature(test)]

extern crate test;

pub fn add_two(a: i32) -> i32 {
    a + 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn it_works() {
        assert_eq!(4, add_two(2));
    }

    #[bench]
    fn bench_add_two(b: &mut Bencher) {
        b.iter(|| add_two(2));
    }
}
```

Note the `test` feature gate, which enables this unstable feature.

We've imported the `test` crate, which contains our benchmarking support.
We have a new function as well, with the `bench` attribute. Unlike regular
tests, which take no arguments, benchmark tests take a `&mut Bencher`. This
`Bencher` provides an `iter` method, which takes a closure. This closure
contains the code we'd like to benchmark.

We can run benchmark tests with `cargo bench`:

```bash
$ cargo bench
   Compiling adder v0.0.1 (file:///home/steve/tmp/adder)
     Running target/release/adder-91b3e234d4ed382a

running 2 tests
test tests::it_works ... ignored
test tests::bench_add_two ... bench:         1 ns/iter (+/- 0)

test result: ok. 0 passed; 0 failed; 1 ignored; 1 measured
```

Our non-benchmark test was ignored. You may have noticed that `cargo bench`
takes a bit longer than `cargo test`. This is because Rust runs our benchmark
a number of times, and then takes the average. Because we're doing so little
work in this example, we have a `1 ns/iter (+/- 0)`, but this would show
the variance if there was one.

Advice on writing benchmarks:


* Move setup code outside the `iter` loop; only put the part you want to measure inside
* Make the code do "the same thing" on each iteration; do not accumulate or change state
* Make the outer function idempotent too; the benchmark runner is likely to run
  it many times
*  Make the inner `iter` loop short and fast so benchmark runs are fast and the
   calibrator can adjust the run-length at fine resolution
* Make the code in the `iter` loop do something simple, to assist in pinpointing
  performance improvements (or regressions)

## Gotcha: optimizations

There's another tricky part to writing benchmarks: benchmarks compiled with
optimizations activated can be dramatically changed by the optimizer so that
the benchmark is no longer benchmarking what one expects. For example, the
compiler might recognize that some calculation has no external effects and
remove it entirely.

```rust,no_run
#![feature(test)]

extern crate test;
use test::Bencher;

#[bench]
fn bench_xor_1000_ints(b: &mut Bencher) {
    b.iter(|| {
        (0..1000).fold(0, |old, new| old ^ new);
    });
}
```

gives the following results

```text
running 1 test
test bench_xor_1000_ints ... bench:         0 ns/iter (+/- 0)

test result: ok. 0 passed; 0 failed; 0 ignored; 1 measured
```

The benchmarking runner offers two ways to avoid this. Either, the closure that
the `iter` method receives can return an arbitrary value which forces the
optimizer to consider the result used and ensures it cannot remove the
computation entirely. This could be done for the example above by adjusting the
`b.iter` call to

```rust
# struct X;
# impl X { fn iter<T, F>(&self, _: F) where F: FnMut() -> T {} } let b = X;
b.iter(|| {
    // Note lack of `;` (could also use an explicit `return`).
    (0..1000).fold(0, |old, new| old ^ new)
});
```

Or, the other option is to call the generic `test::black_box` function, which
is an opaque "black box" to the optimizer and so forces it to consider any
argument as used.

```rust
#![feature(test)]

extern crate test;

# fn main() {
# struct X;
# impl X { fn iter<T, F>(&self, _: F) where F: FnMut() -> T {} } let b = X;
b.iter(|| {
    let n = test::black_box(1000);

    (0..n).fold(0, |a, b| a ^ b)
})
# }
```

Neither of these read or modify the value, and are very cheap for small values.
Larger values can be passed indirectly to reduce overhead (e.g.
`black_box(&huge_struct)`).

Performing either of the above changes gives the following benchmarking results

```text
running 1 test
test bench_xor_1000_ints ... bench:       131 ns/iter (+/- 3)

test result: ok. 0 passed; 0 failed; 0 ignored; 1 measured
```

However, the optimizer can still modify a testcase in an undesirable manner
even when using either of the above.
"##,
    },
    Lint {
        label: "thread_local_internals",
        description: r##"# `thread_local_internals`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "trace_macros",
        description: r##"# `trace_macros`

The tracking issue for this feature is [#29598].

[#29598]: https://github.com/rust-lang/rust/issues/29598

------------------------

With `trace_macros` you can trace the expansion of macros in your code.

## Examples

```rust
#![feature(trace_macros)]

fn main() {
    trace_macros!(true);
    println!("Hello, Rust!");
    trace_macros!(false);
}
```

The `cargo build` output:

```txt
note: trace_macro
 --> src/main.rs:5:5
  |
5 |     println!("Hello, Rust!");
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: expanding `println! { "Hello, Rust!" }`
  = note: to `print ! ( concat ! ( "Hello, Rust!" , "\n" ) )`
  = note: expanding `print! { concat ! ( "Hello, Rust!" , "\n" ) }`
  = note: to `$crate :: io :: _print ( format_args ! ( concat ! ( "Hello, Rust!" , "\n" ) )
          )`

    Finished dev [unoptimized + debuginfo] target(s) in 0.60 secs
```
"##,
    },
    Lint {
        label: "trait_alias",
        description: r##"# `trait_alias`

The tracking issue for this feature is: [#41517]

[#41517]: https://github.com/rust-lang/rust/issues/41517

------------------------

The `trait_alias` feature adds support for trait aliases. These allow aliases
to be created for one or more traits (currently just a single regular trait plus
any number of auto-traits), and used wherever traits would normally be used as
either bounds or trait objects.

```rust
#![feature(trait_alias)]

trait Foo = std::fmt::Debug + Send;
trait Bar = Foo + Sync;

// Use trait alias as bound on type parameter.
fn foo<T: Foo>(v: &T) {
    println!("{:?}", v);
}

pub fn main() {
    foo(&1);

    // Use trait alias for trait objects.
    let a: &Bar = &123;
    println!("{:?}", a);
    let b = Box::new(456) as Box<dyn Foo>;
    println!("{:?}", b);
}
```
"##,
    },
    Lint {
        label: "transparent_unions",
        description: r##"# `transparent_unions`

The tracking issue for this feature is [#60405]

[#60405]: https://github.com/rust-lang/rust/issues/60405

----

The `transparent_unions` feature allows you mark `union`s as
`#[repr(transparent)]`. A `union` may be `#[repr(transparent)]` in exactly the
same conditions in which a `struct` may be `#[repr(transparent)]` (generally,
this means the `union` must have exactly one non-zero-sized field). Some
concrete illustrations follow.

```rust
#![feature(transparent_unions)]

// This union has the same representation as `f32`.
#[repr(transparent)]
union SingleFieldUnion {
    field: f32,
}

// This union has the same representation as `usize`.
#[repr(transparent)]
union MultiFieldUnion {
    field: usize,
    nothing: (),
}
```

For consistency with transparent `struct`s, `union`s must have exactly one
non-zero-sized field. If all fields are zero-sized, the `union` must not be
`#[repr(transparent)]`:

```rust
#![feature(transparent_unions)]

// This (non-transparent) union is already valid in stable Rust:
pub union GoodUnion {
    pub nothing: (),
}

// Error: transparent union needs exactly one non-zero-sized field, but has 0
// #[repr(transparent)]
// pub union BadUnion {
//     pub nothing: (),
// }
```

The one exception is if the `union` is generic over `T` and has a field of type
`T`, it may be `#[repr(transparent)]` even if `T` is a zero-sized type:

```rust
#![feature(transparent_unions)]

// This union has the same representation as `T`.
#[repr(transparent)]
pub union GenericUnion<T: Copy> { // Unions with non-`Copy` fields are unstable.
    pub field: T,
    pub nothing: (),
}

// This is okay even though `()` is a zero-sized type.
pub const THIS_IS_OKAY: GenericUnion<()> = GenericUnion { field: () };
```

Like transarent `struct`s, a transparent `union` of type `U` has the same
layout, size, and ABI as its single non-ZST field. If it is generic over a type
`T`, and all its fields are ZSTs except for exactly one field of type `T`, then
it has the same layout and ABI as `T` (even if `T` is a ZST when monomorphized).

Like transparent `struct`s, transparent `union`s are FFI-safe if and only if
their underlying representation type is also FFI-safe.

A `union` may not be eligible for the same nonnull-style optimizations that a
`struct` or `enum` (with the same fields) are eligible for. Adding
`#[repr(transparent)]` to  `union` does not change this. To give a more concrete
example, it is unspecified whether `size_of::<T>()` is equal to
`size_of::<Option<T>>()`, where `T` is a `union` (regardless of whether or not
it is transparent). The Rust compiler is free to perform this optimization if
possible, but is not required to, and different compiler versions may differ in
their application of these optimizations.
"##,
    },
    Lint {
        label: "try_blocks",
        description: r##"# `try_blocks`

The tracking issue for this feature is: [#31436]

[#31436]: https://github.com/rust-lang/rust/issues/31436

------------------------

The `try_blocks` feature adds support for `try` blocks. A `try`
block creates a new scope one can use the `?` operator in.

```rust,edition2018
#![feature(try_blocks)]

use std::num::ParseIntError;

let result: Result<i32, ParseIntError> = try {
    "1".parse::<i32>()?
        + "2".parse::<i32>()?
        + "3".parse::<i32>()?
};
assert_eq!(result, Ok(6));

let result: Result<i32, ParseIntError> = try {
    "1".parse::<i32>()?
        + "foo".parse::<i32>()?
        + "3".parse::<i32>()?
};
assert!(result.is_err());
```
"##,
    },
    Lint {
        label: "try_trait",
        description: r##"# `try_trait`

The tracking issue for this feature is: [#42327]

[#42327]: https://github.com/rust-lang/rust/issues/42327

------------------------

This introduces a new trait `Try` for extending the `?` operator to types
other than `Result` (a part of [RFC 1859]).  The trait provides the canonical
way to _view_ a type in terms of a success/failure dichotomy.  This will
allow `?` to supplant the `try_opt!` macro on `Option` and the `try_ready!`
macro on `Poll`, among other things.

[RFC 1859]: https://github.com/rust-lang/rfcs/pull/1859

Here's an example implementation of the trait:

```rust,ignore (cannot-reimpl-Try)
/// A distinct type to represent the `None` value of an `Option`.
///
/// This enables using the `?` operator on `Option`; it's rarely useful alone.
#[derive(Debug)]
#[unstable(feature = "try_trait", issue = "42327")]
pub struct None { _priv: () }

#[unstable(feature = "try_trait", issue = "42327")]
impl<T> ops::Try for Option<T>  {
    type Ok = T;
    type Error = None;

    fn into_result(self) -> Result<T, None> {
        self.ok_or(None { _priv: () })
    }

    fn from_ok(v: T) -> Self {
        Some(v)
    }

    fn from_error(_: None) -> Self {
        None
    }
}
```

Note the `Error` associated type here is a new marker.  The `?` operator
allows interconversion between different `Try` implementers only when
the error type can be converted `Into` the error type of the enclosing
function (or catch block).  Having a distinct error type (as opposed to
just `()`, or similar) restricts this to where it's semantically meaningful.
"##,
    },
    Lint {
        label: "unboxed_closures",
        description: r##"# `unboxed_closures`

The tracking issue for this feature is [#29625]

See Also: [`fn_traits`](../library-features/fn-traits.md)

[#29625]: https://github.com/rust-lang/rust/issues/29625

----

The `unboxed_closures` feature allows you to write functions using the `"rust-call"` ABI,
required for implementing the [`Fn*`] family of traits. `"rust-call"` functions must have
exactly one (non self) argument, a tuple representing the argument list.

[`Fn*`]: https://doc.rust-lang.org/std/ops/trait.Fn.html

```rust
#![feature(unboxed_closures)]

extern "rust-call" fn add_args(args: (u32, u32)) -> u32 {
    args.0 + args.1
}

fn main() {}
```
"##,
    },
    Lint {
        label: "unsized_locals",
        description: r##"# `unsized_locals`

The tracking issue for this feature is: [#48055]

[#48055]: https://github.com/rust-lang/rust/issues/48055

------------------------

This implements [RFC1909]. When turned on, you can have unsized arguments and locals:

[RFC1909]: https://github.com/rust-lang/rfcs/blob/master/text/1909-unsized-rvalues.md

```rust
#![allow(incomplete_features)]
#![feature(unsized_locals, unsized_fn_params)]

use std::any::Any;

fn main() {
    let x: Box<dyn Any> = Box::new(42);
    let x: dyn Any = *x;
    //  ^ unsized local variable
    //               ^^ unsized temporary
    foo(x);
}

fn foo(_: dyn Any) {}
//     ^^^^^^ unsized argument
```

The RFC still forbids the following unsized expressions:

```rust,compile_fail
#![feature(unsized_locals)]

use std::any::Any;

struct MyStruct<T: ?Sized> {
    content: T,
}

struct MyTupleStruct<T: ?Sized>(T);

fn answer() -> Box<dyn Any> {
    Box::new(42)
}

fn main() {
    // You CANNOT have unsized statics.
    static X: dyn Any = *answer();  // ERROR
    const Y: dyn Any = *answer();  // ERROR

    // You CANNOT have struct initialized unsized.
    MyStruct { content: *answer() };  // ERROR
    MyTupleStruct(*answer());  // ERROR
    (42, *answer());  // ERROR

    // You CANNOT have unsized return types.
    fn my_function() -> dyn Any { *answer() }  // ERROR

    // You CAN have unsized local variables...
    let mut x: dyn Any = *answer();  // OK
    // ...but you CANNOT reassign to them.
    x = *answer();  // ERROR

    // You CANNOT even initialize them separately.
    let y: dyn Any;  // OK
    y = *answer();  // ERROR

    // Not mentioned in the RFC, but by-move captured variables are also Sized.
    let x: dyn Any = *answer();
    (move || {  // ERROR
        let y = x;
    })();

    // You CAN create a closure with unsized arguments,
    // but you CANNOT call it.
    // This is an implementation detail and may be changed in the future.
    let f = |x: dyn Any| {};
    f(*answer());  // ERROR
}
```

## By-value trait objects

With this feature, you can have by-value `self` arguments without `Self: Sized` bounds.

```rust
#![feature(unsized_fn_params)]

trait Foo {
    fn foo(self) {}
}

impl<T: ?Sized> Foo for T {}

fn main() {
    let slice: Box<[i32]> = Box::new([1, 2, 3]);
    <[i32] as Foo>::foo(*slice);
}
```

And `Foo` will also be object-safe.

```rust
#![feature(unsized_fn_params)]

trait Foo {
    fn foo(self) {}
}

impl<T: ?Sized> Foo for T {}

fn main () {
    let slice: Box<dyn Foo> = Box::new([1, 2, 3]);
    // doesn't compile yet
    <dyn Foo as Foo>::foo(*slice);
}
```

One of the objectives of this feature is to allow `Box<dyn FnOnce>`.

## Variable length arrays

The RFC also describes an extension to the array literal syntax: `[e; dyn n]`. In the syntax, `n` isn't necessarily a constant expression. The array is dynamically allocated on the stack and has the type of `[T]`, instead of `[T; n]`.

```rust,ignore (not-yet-implemented)
#![feature(unsized_locals)]

fn mergesort<T: Ord>(a: &mut [T]) {
    let mut tmp = [T; dyn a.len()];
    // ...
}

fn main() {
    let mut a = [3, 1, 5, 6];
    mergesort(&mut a);
    assert_eq!(a, [1, 3, 5, 6]);
}
```

VLAs are not implemented yet. The syntax isn't final, either. We may need an alternative syntax for Rust 2015 because, in Rust 2015, expressions like `[e; dyn(1)]` would be ambiguous. One possible alternative proposed in the RFC is `[e; n]`: if `n` captures one or more local variables, then it is considered as `[e; dyn n]`.

## Advisory on stack usage

It's advised not to casually use the `#![feature(unsized_locals)]` feature. Typical use-cases are:

- When you need a by-value trait objects.
- When you really need a fast allocation of small temporary arrays.

Another pitfall is repetitive allocation and temporaries. Currently the compiler simply extends the stack frame every time it encounters an unsized assignment. So for example, the code

```rust
#![feature(unsized_locals)]

fn main() {
    let x: Box<[i32]> = Box::new([1, 2, 3, 4, 5]);
    let _x = {{{{{{{{{{*x}}}}}}}}}};
}
```

and the code

```rust
#![feature(unsized_locals)]

fn main() {
    for _ in 0..10 {
        let x: Box<[i32]> = Box::new([1, 2, 3, 4, 5]);
        let _x = *x;
    }
}
```

will unnecessarily extend the stack frame.
"##,
    },
    Lint {
        label: "unsized_tuple_coercion",
        description: r##"# `unsized_tuple_coercion`

The tracking issue for this feature is: [#42877]

[#42877]: https://github.com/rust-lang/rust/issues/42877

------------------------

This is a part of [RFC0401]. According to the RFC, there should be an implementation like this:

```rust,ignore (partial-example)
impl<..., T, U: ?Sized> Unsized<(..., U)> for (..., T) where T: Unsized<U> {}
```

This implementation is currently gated behind `#[feature(unsized_tuple_coercion)]` to avoid insta-stability. Therefore you can use it like this:

```rust
#![feature(unsized_tuple_coercion)]

fn main() {
    let x : ([i32; 3], [i32; 3]) = ([1, 2, 3], [4, 5, 6]);
    let y : &([i32; 3], [i32]) = &x;
    assert_eq!(y.1[0], 4);
}
```

[RFC0401]: https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md
"##,
    },
    Lint {
        label: "update_panic_count",
        description: r##"# `update_panic_count`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "windows_c",
        description: r##"# `windows_c`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "windows_handle",
        description: r##"# `windows_handle`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "windows_net",
        description: r##"# `windows_net`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
    Lint {
        label: "windows_stdio",
        description: r##"# `windows_stdio`

This feature is internal to the Rust compiler and is not intended for general use.

------------------------
"##,
    },
];

pub const CLIPPY_LINTS: &[Lint] = &[
    Lint {
        label: "clippy::absurd_extreme_comparisons",
        description: r##"Checks for comparisons where one side of the relation is
either the minimum or maximum value for its type and warns if it involves a
case that is always true or always false. Only integer and boolean types are
checked."##,
    },
    Lint {
        label: "clippy::almost_swapped",
        description: r##"Checks for `foo = bar; bar = foo` sequences."##,
    },
    Lint {
        label: "clippy::approx_constant",
        description: r##"Checks for floating point literals that approximate
constants which are defined in
[`std::f32::consts`](https://doc.rust-lang.org/stable/std/f32/consts/#constants)
or
[`std::f64::consts`](https://doc.rust-lang.org/stable/std/f64/consts/#constants),
respectively, suggesting to use the predefined constant."##,
    },
    Lint {
        label: "clippy::as_conversions",
        description: r##"Checks for usage of `as` conversions.

Note that this lint is specialized in linting *every single* use of `as`
regardless of whether good alternatives exist or not.
If you want more precise lints for `as`, please consider using these separate lints:
`unnecessary_cast`, `cast_lossless/possible_truncation/possible_wrap/precision_loss/sign_loss`,
`fn_to_numeric_cast(_with_truncation)`, `char_lit_as_u8`, `ref_to_mut` and `ptr_as_ptr`.
There is a good explanation the reason why this lint should work in this way and how it is useful
[in this issue](https://github.com/rust-lang/rust-clippy/issues/5122)."##,
    },
    Lint {
        label: "clippy::assertions_on_constants",
        description: r##"Checks for `assert!(true)` and `assert!(false)` calls."##,
    },
    Lint {
        label: "clippy::assign_op_pattern",
        description: r##"Checks for `a = a op b` or `a = b commutative_op a`
patterns."##,
    },
    Lint {
        label: "clippy::assign_ops",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::async_yields_async",
        description: r##"Checks for async blocks that yield values of types
that can themselves be awaited."##,
    },
    Lint {
        label: "clippy::await_holding_lock",
        description: r##"Checks for calls to await while holding a
non-async-aware MutexGuard."##,
    },
    Lint {
        label: "clippy::await_holding_refcell_ref",
        description: r##"Checks for calls to await while holding a
`RefCell` `Ref` or `RefMut`."##,
    },
    Lint {
        label: "clippy::bad_bit_mask",
        description: r##"Checks for incompatible bit masks in comparisons.

The formula for detecting if an expression of the type `_ <bit_op> m
<cmp_op> c` (where `<bit_op>` is one of {`&`, `|`} and `<cmp_op>` is one of
{`!=`, `>=`, `>`, `!=`, `>=`, `>`}) can be determined from the following
table:

|Comparison  |Bit Op|Example     |is always|Formula               |
|------------|------|------------|---------|----------------------|
|`==` or `!=`| `&`  |`x & 2 == 3`|`false`  |`c & m != c`          |
|`<`  or `>=`| `&`  |`x & 2 < 3` |`true`   |`m < c`               |
|`>`  or `<=`| `&`  |`x & 1 > 1` |`false`  |`m <= c`              |
|`==` or `!=`| `|`  |`x | 1 == 0`|`false`  |`c | m != c`          |
|`<`  or `>=`| `|`  |`x | 1 < 1` |`false`  |`m >= c`              |
|`<=` or `>` | `|`  |`x | 1 > 0` |`true`   |`m > c`               |"##,
    },
    Lint {
        label: "clippy::bind_instead_of_map",
        description: r##"Checks for usage of `_.and_then(|x| Some(y))`, `_.and_then(|x| Ok(y))` or
`_.or_else(|x| Err(y))`."##,
    },
    Lint {
        label: "clippy::blacklisted_name",
        description: r##"Checks for usage of blacklisted names for variables, such
as `foo`."##,
    },
    Lint {
        label: "clippy::blanket_clippy_restriction_lints",
        description: r##"Checks for `warn`/`deny`/`forbid` attributes targeting the whole clippy::restriction category."##,
    },
    Lint {
        label: "clippy::blocks_in_if_conditions",
        description: r##"Checks for `if` conditions that use blocks containing an
expression, statements or conditions that use closures with blocks."##,
    },
    Lint {
        label: "clippy::bool_assert_comparison",
        description: r##"This lint warns about boolean comparisons in assert-like macros."##,
    },
    Lint {
        label: "clippy::bool_comparison",
        description: r##"Checks for expressions of the form `x == true`,
`x != true` and order comparisons such as `x < true` (or vice versa) and
suggest using the variable directly."##,
    },
    Lint {
        label: "clippy::borrow_interior_mutable_const",
        description: r##"Checks if `const` items which is interior mutable (e.g.,
contains a `Cell`, `Mutex`, `AtomicXxxx`, etc.) has been borrowed directly."##,
    },
    Lint {
        label: "clippy::borrowed_box",
        description: r##"Checks for use of `&Box<T>` anywhere in the code.
Check the [Box documentation](https://doc.rust-lang.org/std/boxed/index.html) for more information."##,
    },
    Lint {
        label: "clippy::box_vec",
        description: r##"Checks for use of `Box<Vec<_>>` anywhere in the code.
Check the [Box documentation](https://doc.rust-lang.org/std/boxed/index.html) for more information."##,
    },
    Lint {
        label: "clippy::boxed_local",
        description: r##"Checks for usage of `Box<T>` where an unboxed `T` would
work fine."##,
    },
    Lint {
        label: "clippy::branches_sharing_code",
        description: r##"Checks if the `if` and `else` block contain shared code that can be
moved out of the blocks."##,
    },
    Lint {
        label: "clippy::builtin_type_shadow",
        description: r##"Warns if a generic shadows a built-in type."##,
    },
    Lint {
        label: "clippy::bytes_nth",
        description: r##"Checks for the use of `.bytes().nth()`."##,
    },
    Lint {
        label: "clippy::cargo_common_metadata",
        description: r##"Checks to see if all common metadata is defined in
`Cargo.toml`. See: https://rust-lang-nursery.github.io/api-guidelines/documentation.html#cargotoml-includes-all-common-metadata-c-metadata"##,
    },
    Lint {
        label: "clippy::case_sensitive_file_extension_comparisons",
        description: r##"Checks for calls to `ends_with` with possible file extensions
and suggests to use a case-insensitive approach instead."##,
    },
    Lint {
        label: "clippy::cast_lossless",
        description: r##"Checks for casts between numerical types that may
be replaced by safe conversion functions."##,
    },
    Lint {
        label: "clippy::cast_possible_truncation",
        description: r##"Checks for casts between numerical types that may
truncate large values. This is expected behavior, so the cast is `Allow` by
default."##,
    },
    Lint {
        label: "clippy::cast_possible_wrap",
        description: r##"Checks for casts from an unsigned type to a signed type of
the same size. Performing such a cast is a 'no-op' for the compiler,
i.e., nothing is changed at the bit level, and the binary representation of
the value is reinterpreted. This can cause wrapping if the value is too big
for the target signed type. However, the cast works as defined, so this lint
is `Allow` by default."##,
    },
    Lint {
        label: "clippy::cast_precision_loss",
        description: r##"Checks for casts from any numerical to a float type where
the receiving type cannot store all values from the original type without
rounding errors. This possible rounding is to be expected, so this lint is
`Allow` by default.

Basically, this warns on casting any integer with 32 or more bits to `f32`
or any 64-bit integer to `f64`."##,
    },
    Lint {
        label: "clippy::cast_ptr_alignment",
        description: r##"Checks for casts, using `as` or `pointer::cast`,
from a less-strictly-aligned pointer to a more-strictly-aligned pointer"##,
    },
    Lint {
        label: "clippy::cast_ref_to_mut",
        description: r##"Checks for casts of `&T` to `&mut T` anywhere in the code."##,
    },
    Lint {
        label: "clippy::cast_sign_loss",
        description: r##"Checks for casts from a signed to an unsigned numerical
type. In this case, negative values wrap around to large positive values,
which can be quite surprising in practice. However, as the cast works as
defined, this lint is `Allow` by default."##,
    },
    Lint {
        label: "clippy::char_lit_as_u8",
        description: r##"Checks for expressions where a character literal is cast
to `u8` and suggests using a byte literal instead."##,
    },
    Lint {
        label: "clippy::chars_last_cmp",
        description: r##"Checks for usage of `_.chars().last()` or
`_.chars().next_back()` on a `str` to check if it ends with a given char."##,
    },
    Lint {
        label: "clippy::chars_next_cmp",
        description: r##"Checks for usage of `.chars().next()` on a `str` to check
if it starts with a given char."##,
    },
    Lint {
        label: "clippy::checked_conversions",
        description: r##"Checks for explicit bounds checking when casting."##,
    },
    Lint {
        label: "clippy::clone_double_ref",
        description: r##"Checks for usage of `.clone()` on an `&&T`."##,
    },
    Lint {
        label: "clippy::clone_on_copy",
        description: r##"Checks for usage of `.clone()` on a `Copy` type."##,
    },
    Lint {
        label: "clippy::clone_on_ref_ptr",
        description: r##"Checks for usage of `.clone()` on a ref-counted pointer,
(`Rc`, `Arc`, `rc::Weak`, or `sync::Weak`), and suggests calling Clone via unified
function syntax instead (e.g., `Rc::clone(foo)`)."##,
    },
    Lint {
        label: "clippy::cloned_instead_of_copied",
        description: r##"Checks for usages of `cloned()` on an `Iterator` or `Option` where
`copied()` could be used instead."##,
    },
    Lint { label: "clippy::cmp_nan", description: r##"Checks for comparisons to NaN."## },
    Lint {
        label: "clippy::cmp_null",
        description: r##"This lint checks for equality comparisons with `ptr::null`"##,
    },
    Lint {
        label: "clippy::cmp_owned",
        description: r##"Checks for conversions to owned values just for the sake
of a comparison."##,
    },
    Lint {
        label: "clippy::cognitive_complexity",
        description: r##"Checks for methods with high cognitive complexity."##,
    },
    Lint {
        label: "clippy::collapsible_else_if",
        description: r##"Checks for collapsible `else { if ... }` expressions
that can be collapsed to `else if ...`."##,
    },
    Lint {
        label: "clippy::collapsible_if",
        description: r##"Checks for nested `if` statements which can be collapsed
by `&&`-combining their conditions."##,
    },
    Lint {
        label: "clippy::collapsible_match",
        description: r##"Finds nested `match` or `if let` expressions where the patterns may be collapsed together
without adding any branches.

Note that this lint is not intended to find _all_ cases where nested match patterns can be merged, but only
cases where merging would most likely make the code more readable."##,
    },
    Lint {
        label: "clippy::comparison_chain",
        description: r##"Checks comparison chains written with `if` that can be
rewritten with `match` and `cmp`."##,
    },
    Lint {
        label: "clippy::comparison_to_empty",
        description: r##"Checks for comparing to an empty slice such as `` or `[]`,
and suggests using `.is_empty()` where applicable."##,
    },
    Lint {
        label: "clippy::copy_iterator",
        description: r##"Checks for types that implement `Copy` as well as
`Iterator`."##,
    },
    Lint {
        label: "clippy::create_dir",
        description: r##"Checks usage of `std::fs::create_dir` and suggest using `std::fs::create_dir_all` instead."##,
    },
    Lint {
        label: "clippy::crosspointer_transmute",
        description: r##"Checks for transmutes between a type `T` and `*T`."##,
    },
    Lint { label: "clippy::dbg_macro", description: r##"Checks for usage of dbg!() macro."## },
    Lint {
        label: "clippy::debug_assert_with_mut_call",
        description: r##"Checks for function/method calls with a mutable
parameter in `debug_assert!`, `debug_assert_eq!` and `debug_assert_ne!` macros."##,
    },
    Lint {
        label: "clippy::decimal_literal_representation",
        description: r##"Warns if there is a better representation for a numeric literal."##,
    },
    Lint {
        label: "clippy::declare_interior_mutable_const",
        description: r##"Checks for declaration of `const` items which is interior
mutable (e.g., contains a `Cell`, `Mutex`, `AtomicXxxx`, etc.)."##,
    },
    Lint {
        label: "clippy::default_numeric_fallback",
        description: r##"Checks for usage of unconstrained numeric literals which may cause default numeric fallback in type
inference.

Default numeric fallback means that if numeric types have not yet been bound to concrete
types at the end of type inference, then integer type is bound to `i32`, and similarly
floating type is bound to `f64`.

See [RFC0212](https://github.com/rust-lang/rfcs/blob/master/text/0212-restore-int-fallback.md) for more information about the fallback."##,
    },
    Lint {
        label: "clippy::default_trait_access",
        description: r##"Checks for literal calls to `Default::default()`."##,
    },
    Lint {
        label: "clippy::deprecated_cfg_attr",
        description: r##"Checks for `#[cfg_attr(rustfmt, rustfmt_skip)]` and suggests to replace it
with `#[rustfmt::skip]`."##,
    },
    Lint {
        label: "clippy::deprecated_semver",
        description: r##"Checks for `#[deprecated]` annotations with a `since`
field that is not a valid semantic version."##,
    },
    Lint {
        label: "clippy::deref_addrof",
        description: r##"Checks for usage of `*&` and `*&mut` in expressions."##,
    },
    Lint {
        label: "clippy::derive_hash_xor_eq",
        description: r##"Checks for deriving `Hash` but implementing `PartialEq`
explicitly or vice versa."##,
    },
    Lint {
        label: "clippy::derive_ord_xor_partial_ord",
        description: r##"Checks for deriving `Ord` but implementing `PartialOrd`
explicitly or vice versa."##,
    },
    Lint {
        label: "clippy::disallowed_method",
        description: r##"Denies the configured methods and functions in clippy.toml"##,
    },
    Lint {
        label: "clippy::diverging_sub_expression",
        description: r##"Checks for diverging calls that are not match arms or
statements."##,
    },
    Lint {
        label: "clippy::doc_markdown",
        description: r##"Checks for the presence of `_`, `::` or camel-case words
outside ticks in documentation."##,
    },
    Lint {
        label: "clippy::double_comparisons",
        description: r##"Checks for double comparisons that could be simplified to a single expression."##,
    },
    Lint {
        label: "clippy::double_must_use",
        description: r##"Checks for a [`#[must_use]`] attribute without
further information on functions and methods that return a type already
marked as `#[must_use]`.

[`#[must_use]`]: https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute"##,
    },
    Lint {
        label: "clippy::double_neg",
        description: r##"Detects expressions of the form `--x`."##,
    },
    Lint {
        label: "clippy::double_parens",
        description: r##"Checks for unnecessary double parentheses."##,
    },
    Lint {
        label: "clippy::drop_copy",
        description: r##"Checks for calls to `std::mem::drop` with a value
that derives the Copy trait"##,
    },
    Lint {
        label: "clippy::drop_ref",
        description: r##"Checks for calls to `std::mem::drop` with a reference
instead of an owned value."##,
    },
    Lint {
        label: "clippy::duplicate_underscore_argument",
        description: r##"Checks for function arguments having the similar names
differing by an underscore."##,
    },
    Lint {
        label: "clippy::duration_subsec",
        description: r##"Checks for calculation of subsecond microseconds or milliseconds
from other `Duration` methods."##,
    },
    Lint {
        label: "clippy::else_if_without_else",
        description: r##"Checks for usage of if expressions with an `else if` branch,
but without a final `else` branch."##,
    },
    Lint {
        label: "clippy::empty_enum",
        description: r##"Checks for `enum`s with no variants.

As of this writing, the `never_type` is still a
nightly-only experimental API. Therefore, this lint is only triggered
if the `never_type` is enabled."##,
    },
    Lint {
        label: "clippy::empty_line_after_outer_attr",
        description: r##"Checks for empty lines after outer attributes"##,
    },
    Lint { label: "clippy::empty_loop", description: r##"Checks for empty `loop` expressions."## },
    Lint {
        label: "clippy::enum_clike_unportable_variant",
        description: r##"Checks for C-like enumerations that are
`repr(isize/usize)` and have values that don't fit into an `i32`."##,
    },
    Lint { label: "clippy::enum_glob_use", description: r##"Checks for `use Enum::*`."## },
    Lint {
        label: "clippy::enum_variant_names",
        description: r##"Detects enumeration variants that are prefixed or suffixed
by the same characters."##,
    },
    Lint {
        label: "clippy::eq_op",
        description: r##"Checks for equal operands to comparison, logical and
bitwise, difference and division binary operators (`==`, `>`, etc., `&&`,
`||`, `&`, `|`, `^`, `-` and `/`)."##,
    },
    Lint {
        label: "clippy::erasing_op",
        description: r##"Checks for erasing operations, e.g., `x * 0`."##,
    },
    Lint {
        label: "clippy::eval_order_dependence",
        description: r##"Checks for a read and a write to the same variable where
whether the read occurs before or after the write depends on the evaluation
order of sub-expressions."##,
    },
    Lint {
        label: "clippy::excessive_precision",
        description: r##"Checks for float literals with a precision greater
than that supported by the underlying type."##,
    },
    Lint {
        label: "clippy::exhaustive_enums",
        description: r##"Warns on any exported `enum`s that are not tagged `#[non_exhaustive]`"##,
    },
    Lint {
        label: "clippy::exhaustive_structs",
        description: r##"Warns on any exported `structs`s that are not tagged `#[non_exhaustive]`"##,
    },
    Lint {
        label: "clippy::exit",
        description: r##"`exit()`  terminates the program and doesn't provide a
stack trace."##,
    },
    Lint {
        label: "clippy::expect_fun_call",
        description: r##"Checks for calls to `.expect(&format!(...))`, `.expect(foo(..))`,
etc., and suggests to use `unwrap_or_else` instead"##,
    },
    Lint {
        label: "clippy::expect_used",
        description: r##"Checks for `.expect()` calls on `Option`s and `Result`s."##,
    },
    Lint {
        label: "clippy::expl_impl_clone_on_copy",
        description: r##"Checks for explicit `Clone` implementations for `Copy`
types."##,
    },
    Lint {
        label: "clippy::explicit_counter_loop",
        description: r##"Checks `for` loops over slices with an explicit counter
and suggests the use of `.enumerate()`."##,
    },
    Lint {
        label: "clippy::explicit_deref_methods",
        description: r##"Checks for explicit `deref()` or `deref_mut()` method calls."##,
    },
    Lint {
        label: "clippy::explicit_into_iter_loop",
        description: r##"Checks for loops on `y.into_iter()` where `y` will do, and
suggests the latter."##,
    },
    Lint {
        label: "clippy::explicit_iter_loop",
        description: r##"Checks for loops on `x.iter()` where `&x` will do, and
suggests the latter."##,
    },
    Lint {
        label: "clippy::explicit_write",
        description: r##"Checks for usage of `write!()` / `writeln()!` which can be
replaced with `(e)print!()` / `(e)println!()`"##,
    },
    Lint {
        label: "clippy::extend_from_slice",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::extra_unused_lifetimes",
        description: r##"Checks for lifetimes in generics that are never used
anywhere else."##,
    },
    Lint {
        label: "clippy::fallible_impl_from",
        description: r##"Checks for impls of `From<..>` that contain `panic!()` or `unwrap()`"##,
    },
    Lint {
        label: "clippy::field_reassign_with_default",
        description: r##"Checks for immediate reassignment of fields initialized
with Default::default()."##,
    },
    Lint {
        label: "clippy::filetype_is_file",
        description: r##"Checks for `FileType::is_file()`."##,
    },
    Lint {
        label: "clippy::filter_map",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::filter_map_identity",
        description: r##"Checks for usage of `filter_map(|x| x)`."##,
    },
    Lint {
        label: "clippy::filter_map_next",
        description: r##"Checks for usage of `_.filter_map(_).next()`."##,
    },
    Lint {
        label: "clippy::filter_next",
        description: r##"Checks for usage of `_.filter(_).next()`."##,
    },
    Lint { label: "clippy::find_map", description: r##"Nothing. This lint has been deprecated."## },
    Lint {
        label: "clippy::flat_map_identity",
        description: r##"Checks for usage of `flat_map(|x| x)`."##,
    },
    Lint {
        label: "clippy::flat_map_option",
        description: r##"Checks for usages of `Iterator::flat_map()` where `filter_map()` could be
used instead."##,
    },
    Lint { label: "clippy::float_arithmetic", description: r##"Checks for float arithmetic."## },
    Lint {
        label: "clippy::float_cmp",
        description: r##"Checks for (in-)equality comparisons on floating-point
values (apart from zero), except in functions called `*eq*` (which probably
implement equality for a type involving floats)."##,
    },
    Lint {
        label: "clippy::float_cmp_const",
        description: r##"Checks for (in-)equality comparisons on floating-point
value and constant, except in functions called `*eq*` (which probably
implement equality for a type involving floats)."##,
    },
    Lint {
        label: "clippy::float_equality_without_abs",
        description: r##"Checks for statements of the form `(a - b) < f32::EPSILON` or
`(a - b) < f64::EPSILON`. Notes the missing `.abs()`."##,
    },
    Lint {
        label: "clippy::fn_address_comparisons",
        description: r##"Checks for comparisons with an address of a function item."##,
    },
    Lint {
        label: "clippy::fn_params_excessive_bools",
        description: r##"Checks for excessive use of
bools in function definitions."##,
    },
    Lint {
        label: "clippy::fn_to_numeric_cast",
        description: r##"Checks for casts of function pointers to something other than usize"##,
    },
    Lint {
        label: "clippy::fn_to_numeric_cast_with_truncation",
        description: r##"Checks for casts of a function pointer to a numeric type not wide enough to
store address."##,
    },
    Lint {
        label: "clippy::for_kv_map",
        description: r##"Checks for iterating a map (`HashMap` or `BTreeMap`) and
ignoring either the keys or values."##,
    },
    Lint {
        label: "clippy::for_loops_over_fallibles",
        description: r##"Checks for `for` loops over `Option` or `Result` values."##,
    },
    Lint {
        label: "clippy::forget_copy",
        description: r##"Checks for calls to `std::mem::forget` with a value that
derives the Copy trait"##,
    },
    Lint {
        label: "clippy::forget_ref",
        description: r##"Checks for calls to `std::mem::forget` with a reference
instead of an owned value."##,
    },
    Lint {
        label: "clippy::from_iter_instead_of_collect",
        description: r##"Checks for `from_iter()` function calls on types that implement the `FromIterator`
trait."##,
    },
    Lint {
        label: "clippy::from_over_into",
        description: r##"Searches for implementations of the `Into<..>` trait and suggests to implement `From<..>` instead."##,
    },
    Lint {
        label: "clippy::from_str_radix_10",
        description: r##"Checks for function invocations of the form `primitive::from_str_radix(s, 10)`"##,
    },
    Lint {
        label: "clippy::future_not_send",
        description: r##"This lint requires Future implementations returned from
functions and methods to implement the `Send` marker trait. It is mostly
used by library authors (public and internal) that target an audience where
multithreaded executors are likely to be used for running these Futures."##,
    },
    Lint {
        label: "clippy::get_last_with_len",
        description: r##"Checks for using `x.get(x.len() - 1)` instead of
`x.last()`."##,
    },
    Lint {
        label: "clippy::get_unwrap",
        description: r##"Checks for use of `.get().unwrap()` (or
`.get_mut().unwrap`) on a standard library type which implements `Index`"##,
    },
    Lint {
        label: "clippy::identity_op",
        description: r##"Checks for identity operations, e.g., `x + 0`."##,
    },
    Lint {
        label: "clippy::if_let_mutex",
        description: r##"Checks for `Mutex::lock` calls in `if let` expression
with lock calls in any of the else blocks."##,
    },
    Lint {
        label: "clippy::if_let_redundant_pattern_matching",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::if_let_some_result",
        description: r##"* Checks for unnecessary `ok()` in if let."##,
    },
    Lint {
        label: "clippy::if_not_else",
        description: r##"Checks for usage of `!` or `!=` in an if condition with an
else branch."##,
    },
    Lint {
        label: "clippy::if_same_then_else",
        description: r##"Checks for `if/else` with the same body as the *then* part
and the *else* part."##,
    },
    Lint {
        label: "clippy::if_then_some_else_none",
        description: r##"Checks for if-else that could be written to `bool::then`."##,
    },
    Lint {
        label: "clippy::ifs_same_cond",
        description: r##"Checks for consecutive `if`s with the same condition."##,
    },
    Lint {
        label: "clippy::implicit_clone",
        description: r##"Checks for the usage of `_.to_owned()`, `vec.to_vec()`, or similar when calling `_.clone()` would be clearer."##,
    },
    Lint {
        label: "clippy::implicit_hasher",
        description: r##"Checks for public `impl` or `fn` missing generalization
over different hashers and implicitly defaulting to the default hashing
algorithm (`SipHash`)."##,
    },
    Lint {
        label: "clippy::implicit_return",
        description: r##"Checks for missing return statements at the end of a block."##,
    },
    Lint {
        label: "clippy::implicit_saturating_sub",
        description: r##"Checks for implicit saturating subtraction."##,
    },
    Lint {
        label: "clippy::imprecise_flops",
        description: r##"Looks for floating-point expressions that
can be expressed using built-in methods to improve accuracy
at the cost of performance."##,
    },
    Lint {
        label: "clippy::inconsistent_digit_grouping",
        description: r##"Warns if an integral or floating-point constant is
grouped inconsistently with underscores."##,
    },
    Lint {
        label: "clippy::inconsistent_struct_constructor",
        description: r##"Checks for struct constructors where all fields are shorthand and
the order of the field init shorthand in the constructor is inconsistent
with the order in the struct definition."##,
    },
    Lint {
        label: "clippy::indexing_slicing",
        description: r##"Checks for usage of indexing or slicing. Arrays are special cases, this lint
does report on arrays if we can tell that slicing operations are in bounds and does not
lint on constant `usize` indexing on arrays because that is handled by rustc's `const_err` lint."##,
    },
    Lint {
        label: "clippy::ineffective_bit_mask",
        description: r##"Checks for bit masks in comparisons which can be removed
without changing the outcome. The basic structure can be seen in the
following table:

|Comparison| Bit Op  |Example    |equals |
|----------|---------|-----------|-------|
|`>` / `<=`|`|` / `^`|`x | 2 > 3`|`x > 3`|
|`<` / `>=`|`|` / `^`|`x ^ 1 < 4`|`x < 4`|"##,
    },
    Lint {
        label: "clippy::inefficient_to_string",
        description: r##"Checks for usage of `.to_string()` on an `&&T` where
`T` implements `ToString` directly (like `&&str` or `&&String`)."##,
    },
    Lint {
        label: "clippy::infallible_destructuring_match",
        description: r##"Checks for matches being used to destructure a single-variant enum
or tuple struct where a `let` will suffice."##,
    },
    Lint {
        label: "clippy::infinite_iter",
        description: r##"Checks for iteration that is guaranteed to be infinite."##,
    },
    Lint {
        label: "clippy::inherent_to_string",
        description: r##"Checks for the definition of inherent methods with a signature of `to_string(&self) -> String`."##,
    },
    Lint {
        label: "clippy::inherent_to_string_shadow_display",
        description: r##"Checks for the definition of inherent methods with a signature of `to_string(&self) -> String` and if the type implementing this method also implements the `Display` trait."##,
    },
    Lint {
        label: "clippy::inline_always",
        description: r##"Checks for items annotated with `#[inline(always)]`,
unless the annotated function is empty or simply panics."##,
    },
    Lint {
        label: "clippy::inline_asm_x86_att_syntax",
        description: r##"Checks for usage of AT&T x86 assembly syntax."##,
    },
    Lint {
        label: "clippy::inline_asm_x86_intel_syntax",
        description: r##"Checks for usage of Intel x86 assembly syntax."##,
    },
    Lint {
        label: "clippy::inline_fn_without_body",
        description: r##"Checks for `#[inline]` on trait methods without bodies"##,
    },
    Lint {
        label: "clippy::inspect_for_each",
        description: r##"Checks for usage of `inspect().for_each()`."##,
    },
    Lint {
        label: "clippy::int_plus_one",
        description: r##"Checks for usage of `x >= y + 1` or `x - 1 >= y` (and `<=`) in a block"##,
    },
    Lint {
        label: "clippy::integer_arithmetic",
        description: r##"Checks for integer arithmetic operations which could overflow or panic.

Specifically, checks for any operators (`+`, `-`, `*`, `<<`, etc) which are capable
of overflowing according to the [Rust
Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#overflow),
or which can panic (`/`, `%`). No bounds analysis or sophisticated reasoning is
attempted."##,
    },
    Lint { label: "clippy::integer_division", description: r##"Checks for division of integers"## },
    Lint {
        label: "clippy::into_iter_on_ref",
        description: r##"Checks for `into_iter` calls on references which should be replaced by `iter`
or `iter_mut`."##,
    },
    Lint {
        label: "clippy::invalid_atomic_ordering",
        description: r##"Checks for usage of invalid atomic
ordering in atomic loads/stores/exchanges/updates and
memory fences."##,
    },
    Lint {
        label: "clippy::invalid_null_ptr_usage",
        description: r##"This lint checks for invalid usages of `ptr::null`."##,
    },
    Lint {
        label: "clippy::invalid_regex",
        description: r##"Checks [regex](https://crates.io/crates/regex) creation
(with `Regex::new`, `RegexBuilder::new`, or `RegexSet::new`) for correct
regex syntax."##,
    },
    Lint {
        label: "clippy::invalid_upcast_comparisons",
        description: r##"Checks for comparisons where the relation is always either
true or false, but where one side has been upcast so that the comparison is
necessary. Only integer types are checked."##,
    },
    Lint {
        label: "clippy::invisible_characters",
        description: r##"Checks for invisible Unicode characters in the code."##,
    },
    Lint {
        label: "clippy::items_after_statements",
        description: r##"Checks for items declared after some statement in a block."##,
    },
    Lint {
        label: "clippy::iter_cloned_collect",
        description: r##"Checks for the use of `.cloned().collect()` on slice to
create a `Vec`."##,
    },
    Lint {
        label: "clippy::iter_count",
        description: r##"Checks for the use of `.iter().count()`."##,
    },
    Lint { label: "clippy::iter_next_loop", description: r##"Checks for loops on `x.next()`."## },
    Lint {
        label: "clippy::iter_next_slice",
        description: r##"Checks for usage of `iter().next()` on a Slice or an Array"##,
    },
    Lint {
        label: "clippy::iter_nth",
        description: r##"Checks for use of `.iter().nth()` (and the related
`.iter_mut().nth()`) on standard library types with O(1) element access."##,
    },
    Lint {
        label: "clippy::iter_nth_zero",
        description: r##"Checks for the use of `iter.nth(0)`."##,
    },
    Lint {
        label: "clippy::iter_skip_next",
        description: r##"Checks for use of `.skip(x).next()` on iterators."##,
    },
    Lint {
        label: "clippy::iterator_step_by_zero",
        description: r##"Checks for calling `.step_by(0)` on iterators which panics."##,
    },
    Lint {
        label: "clippy::just_underscores_and_digits",
        description: r##"Checks if you have variables whose name consists of just
underscores and digits."##,
    },
    Lint {
        label: "clippy::large_const_arrays",
        description: r##"Checks for large `const` arrays that should
be defined as `static` instead."##,
    },
    Lint {
        label: "clippy::large_digit_groups",
        description: r##"Warns if the digits of an integral or floating-point
constant are grouped into groups that
are too large."##,
    },
    Lint {
        label: "clippy::large_enum_variant",
        description: r##"Checks for large size differences between variants on
`enum`s."##,
    },
    Lint {
        label: "clippy::large_stack_arrays",
        description: r##"Checks for local arrays that may be too large."##,
    },
    Lint {
        label: "clippy::large_types_passed_by_value",
        description: r##"Checks for functions taking arguments by value, where
the argument type is `Copy` and large enough to be worth considering
passing by reference. Does not trigger if the function is being exported,
because that might induce API breakage, if the parameter is declared as mutable,
or if the argument is a `self`."##,
    },
    Lint {
        label: "clippy::len_without_is_empty",
        description: r##"Checks for items that implement `.len()` but not
`.is_empty()`."##,
    },
    Lint {
        label: "clippy::len_zero",
        description: r##"Checks for getting the length of something via `.len()`
just to compare to zero, and suggests using `.is_empty()` where applicable."##,
    },
    Lint {
        label: "clippy::let_and_return",
        description: r##"Checks for `let`-bindings, which are subsequently
returned."##,
    },
    Lint {
        label: "clippy::let_underscore_drop",
        description: r##"Checks for `let _ = <expr>`
where expr has a type that implements `Drop`"##,
    },
    Lint {
        label: "clippy::let_underscore_lock",
        description: r##"Checks for `let _ = sync_lock`"##,
    },
    Lint {
        label: "clippy::let_underscore_must_use",
        description: r##"Checks for `let _ = <expr>`
where expr is #[must_use]"##,
    },
    Lint { label: "clippy::let_unit_value", description: r##"Checks for binding a unit value."## },
    Lint {
        label: "clippy::linkedlist",
        description: r##"Checks for usage of any `LinkedList`, suggesting to use a
`Vec` or a `VecDeque` (formerly called `RingBuf`)."##,
    },
    Lint {
        label: "clippy::logic_bug",
        description: r##"Checks for boolean expressions that contain terminals that
can be eliminated."##,
    },
    Lint {
        label: "clippy::lossy_float_literal",
        description: r##"Checks for whole number float literals that
cannot be represented as the underlying type without loss."##,
    },
    Lint {
        label: "clippy::macro_use_imports",
        description: r##"Checks for `#[macro_use] use...`."##,
    },
    Lint {
        label: "clippy::main_recursion",
        description: r##"Checks for recursion using the entrypoint."##,
    },
    Lint {
        label: "clippy::manual_async_fn",
        description: r##"It checks for manual implementations of `async` functions."##,
    },
    Lint {
        label: "clippy::manual_filter_map",
        description: r##"Checks for usage of `_.filter(_).map(_)` that can be written more simply
as `filter_map(_)`."##,
    },
    Lint {
        label: "clippy::manual_find_map",
        description: r##"Checks for usage of `_.find(_).map(_)` that can be written more simply
as `find_map(_)`."##,
    },
    Lint {
        label: "clippy::manual_flatten",
        description: r##"Check for unnecessary `if let` usage in a for loop
where only the `Some` or `Ok` variant of the iterator element is used."##,
    },
    Lint {
        label: "clippy::manual_map",
        description: r##"Checks for usages of `match` which could be implemented using `map`"##,
    },
    Lint {
        label: "clippy::manual_memcpy",
        description: r##"Checks for for-loops that manually copy items between
slices that could be optimized by having a memcpy."##,
    },
    Lint {
        label: "clippy::manual_non_exhaustive",
        description: r##"Checks for manual implementations of the non-exhaustive pattern."##,
    },
    Lint {
        label: "clippy::manual_ok_or",
        description: r##"Finds patterns that reimplement `Option::ok_or`."##,
    },
    Lint {
        label: "clippy::manual_range_contains",
        description: r##"Checks for expressions like `x >= 3 && x < 8` that could
be more readably expressed as `(3..8).contains(x)`."##,
    },
    Lint {
        label: "clippy::manual_saturating_arithmetic",
        description: r##"Checks for `.checked_add/sub(x).unwrap_or(MAX/MIN)`."##,
    },
    Lint {
        label: "clippy::manual_str_repeat",
        description: r##"Checks for manual implementations of `str::repeat`"##,
    },
    Lint {
        label: "clippy::manual_strip",
        description: r##"Suggests using `strip_{prefix,suffix}` over `str::{starts,ends}_with` and slicing using
the pattern's length."##,
    },
    Lint { label: "clippy::manual_swap", description: r##"Checks for manual swapping."## },
    Lint {
        label: "clippy::manual_unwrap_or",
        description: r##"Finds patterns that reimplement `Option::unwrap_or` or `Result::unwrap_or`."##,
    },
    Lint {
        label: "clippy::many_single_char_names",
        description: r##"Checks for too many variables whose name consists of a
single character."##,
    },
    Lint {
        label: "clippy::map_clone",
        description: r##"Checks for usage of `map(|x| x.clone())` or
dereferencing closures for `Copy` types, on `Iterator` or `Option`,
and suggests `cloned()` or `copied()` instead"##,
    },
    Lint {
        label: "clippy::map_collect_result_unit",
        description: r##"Checks for usage of `_.map(_).collect::<Result<(), _>()`."##,
    },
    Lint {
        label: "clippy::map_entry",
        description: r##"Checks for uses of `contains_key` + `insert` on `HashMap`
or `BTreeMap`."##,
    },
    Lint {
        label: "clippy::map_err_ignore",
        description: r##"Checks for instances of `map_err(|_| Some::Enum)`"##,
    },
    Lint {
        label: "clippy::map_flatten",
        description: r##"Checks for usage of `_.map(_).flatten(_)` on `Iterator` and `Option`"##,
    },
    Lint {
        label: "clippy::map_identity",
        description: r##"Checks for instances of `map(f)` where `f` is the identity function."##,
    },
    Lint {
        label: "clippy::map_unwrap_or",
        description: r##"Checks for usage of `option.map(_).unwrap_or(_)` or `option.map(_).unwrap_or_else(_)` or
`result.map(_).unwrap_or_else(_)`."##,
    },
    Lint {
        label: "clippy::match_as_ref",
        description: r##"Checks for match which is used to add a reference to an
`Option` value."##,
    },
    Lint {
        label: "clippy::match_bool",
        description: r##"Checks for matches where match expression is a `bool`. It
suggests to replace the expression with an `if...else` block."##,
    },
    Lint {
        label: "clippy::match_like_matches_macro",
        description: r##"Checks for `match`  or `if let` expressions producing a
`bool` that could be written using `matches!`"##,
    },
    Lint {
        label: "clippy::match_on_vec_items",
        description: r##"Checks for `match vec[idx]` or `match vec[n..m]`."##,
    },
    Lint {
        label: "clippy::match_overlapping_arm",
        description: r##"Checks for overlapping match arms."##,
    },
    Lint {
        label: "clippy::match_ref_pats",
        description: r##"Checks for matches where all arms match a reference,
suggesting to remove the reference and deref the matched expression
instead. It also checks for `if let &foo = bar` blocks."##,
    },
    Lint {
        label: "clippy::match_same_arms",
        description: r##"Checks for `match` with identical arm bodies."##,
    },
    Lint {
        label: "clippy::match_single_binding",
        description: r##"Checks for useless match that binds to only one value."##,
    },
    Lint {
        label: "clippy::match_wild_err_arm",
        description: r##"Checks for arm which matches all errors with `Err(_)`
and take drastic actions like `panic!`."##,
    },
    Lint {
        label: "clippy::match_wildcard_for_single_variants",
        description: r##"Checks for wildcard enum matches for a single variant."##,
    },
    Lint {
        label: "clippy::maybe_infinite_iter",
        description: r##"Checks for iteration that may be infinite."##,
    },
    Lint {
        label: "clippy::mem_discriminant_non_enum",
        description: r##"Checks for calls of `mem::discriminant()` on a non-enum type."##,
    },
    Lint {
        label: "clippy::mem_forget",
        description: r##"Checks for usage of `std::mem::forget(t)` where `t` is
`Drop`."##,
    },
    Lint {
        label: "clippy::mem_replace_option_with_none",
        description: r##"Checks for `mem::replace()` on an `Option` with
`None`."##,
    },
    Lint {
        label: "clippy::mem_replace_with_default",
        description: r##"Checks for `std::mem::replace` on a value of type
`T` with `T::default()`."##,
    },
    Lint {
        label: "clippy::mem_replace_with_uninit",
        description: r##"Checks for `mem::replace(&mut _, mem::uninitialized())`
and `mem::replace(&mut _, mem::zeroed())`."##,
    },
    Lint {
        label: "clippy::min_max",
        description: r##"Checks for expressions where `std::cmp::min` and `max` are
used to clamp values, but switched so that the result is constant."##,
    },
    Lint {
        label: "clippy::misaligned_transmute",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::mismatched_target_os",
        description: r##"Checks for cfg attributes having operating systems used in target family position."##,
    },
    Lint {
        label: "clippy::misrefactored_assign_op",
        description: r##"Checks for `a op= a op b` or `a op= b op a` patterns."##,
    },
    Lint {
        label: "clippy::missing_const_for_fn",
        description: r##"Suggests the use of `const` in functions and methods where possible."##,
    },
    Lint {
        label: "clippy::missing_docs_in_private_items",
        description: r##"Warns if there is missing doc for any documentable item
(public or private)."##,
    },
    Lint {
        label: "clippy::missing_errors_doc",
        description: r##"Checks the doc comments of publicly visible functions that
return a `Result` type and warns if there is no `# Errors` section."##,
    },
    Lint {
        label: "clippy::missing_inline_in_public_items",
        description: r##"it lints if an exported function, method, trait method with default impl,
or trait method impl is not `#[inline]`."##,
    },
    Lint {
        label: "clippy::missing_panics_doc",
        description: r##"Checks the doc comments of publicly visible functions that
may panic and warns if there is no `# Panics` section."##,
    },
    Lint {
        label: "clippy::missing_safety_doc",
        description: r##"Checks for the doc comments of publicly visible
unsafe functions and warns if there is no `# Safety` section."##,
    },
    Lint {
        label: "clippy::mistyped_literal_suffixes",
        description: r##"Warns for mistyped suffix in literals"##,
    },
    Lint {
        label: "clippy::mixed_case_hex_literals",
        description: r##"Warns on hexadecimal literals with mixed-case letter
digits."##,
    },
    Lint {
        label: "clippy::module_inception",
        description: r##"Checks for modules that have the same name as their
parent module"##,
    },
    Lint {
        label: "clippy::module_name_repetitions",
        description: r##"Detects type names that are prefixed or suffixed by the
containing module's name."##,
    },
    Lint { label: "clippy::modulo_arithmetic", description: r##"Checks for modulo arithmetic."## },
    Lint {
        label: "clippy::modulo_one",
        description: r##"Checks for getting the remainder of a division by one or minus
one."##,
    },
    Lint {
        label: "clippy::multiple_crate_versions",
        description: r##"Checks to see if multiple versions of a crate are being
used."##,
    },
    Lint {
        label: "clippy::multiple_inherent_impl",
        description: r##"Checks for multiple inherent implementations of a struct"##,
    },
    Lint {
        label: "clippy::must_use_candidate",
        description: r##"Checks for public functions that have no
[`#[must_use]`] attribute, but return something not already marked
must-use, have no mutable arg and mutate no statics.

[`#[must_use]`]: https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute"##,
    },
    Lint {
        label: "clippy::must_use_unit",
        description: r##"Checks for a [`#[must_use]`] attribute on
unit-returning functions and methods.

[`#[must_use]`]: https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute"##,
    },
    Lint {
        label: "clippy::mut_from_ref",
        description: r##"This lint checks for functions that take immutable
references and return mutable ones."##,
    },
    Lint {
        label: "clippy::mut_mut",
        description: r##"Checks for instances of `mut mut` references."##,
    },
    Lint {
        label: "clippy::mut_mutex_lock",
        description: r##"Checks for `&mut Mutex::lock` calls"##,
    },
    Lint {
        label: "clippy::mut_range_bound",
        description: r##"Checks for loops which have a range bound that is a mutable variable"##,
    },
    Lint {
        label: "clippy::mutable_key_type",
        description: r##"Checks for sets/maps with mutable key types."##,
    },
    Lint {
        label: "clippy::mutex_atomic",
        description: r##"Checks for usages of `Mutex<X>` where an atomic will do."##,
    },
    Lint {
        label: "clippy::mutex_integer",
        description: r##"Checks for usages of `Mutex<X>` where `X` is an integral
type."##,
    },
    Lint { label: "clippy::naive_bytecount", description: r##"Checks for naive byte counts"## },
    Lint {
        label: "clippy::needless_arbitrary_self_type",
        description: r##"The lint checks for `self` in fn parameters that
specify the `Self`-type explicitly"##,
    },
    Lint {
        label: "clippy::needless_bitwise_bool",
        description: r##"Checks for uses of bitwise and/or operators between booleans, where performance may be improved by using
a lazy and."##,
    },
    Lint {
        label: "clippy::needless_bool",
        description: r##"Checks for expressions of the form `if c { true } else {
false }` (or vice versa) and suggests using the condition directly."##,
    },
    Lint {
        label: "clippy::needless_borrow",
        description: r##"Checks for address of operations (`&`) that are going to
be dereferenced immediately by the compiler."##,
    },
    Lint {
        label: "clippy::needless_borrowed_reference",
        description: r##"Checks for bindings that destructure a reference and borrow the inner
value with `&ref`."##,
    },
    Lint {
        label: "clippy::needless_collect",
        description: r##"Checks for functions collecting an iterator when collect
is not needed."##,
    },
    Lint {
        label: "clippy::needless_continue",
        description: r##"The lint checks for `if`-statements appearing in loops
that contain a `continue` statement in either their main blocks or their
`else`-blocks, when omitting the `else`-block possibly with some
rearrangement of code can make the code easier to understand."##,
    },
    Lint {
        label: "clippy::needless_doctest_main",
        description: r##"Checks for `fn main() { .. }` in doctests"##,
    },
    Lint {
        label: "clippy::needless_for_each",
        description: r##"Checks for usage of `for_each` that would be more simply written as a
`for` loop."##,
    },
    Lint {
        label: "clippy::needless_lifetimes",
        description: r##"Checks for lifetime annotations which can be removed by
relying on lifetime elision."##,
    },
    Lint {
        label: "clippy::needless_pass_by_value",
        description: r##"Checks for functions taking arguments by value, but not
consuming them in its
body."##,
    },
    Lint {
        label: "clippy::needless_question_mark",
        description: r##"Suggests alternatives for useless applications of `?` in terminating expressions"##,
    },
    Lint {
        label: "clippy::needless_range_loop",
        description: r##"Checks for looping over the range of `0..len` of some
collection just to get the values by index."##,
    },
    Lint {
        label: "clippy::needless_return",
        description: r##"Checks for return statements at the end of a block."##,
    },
    Lint {
        label: "clippy::needless_update",
        description: r##"Checks for needlessly including a base struct on update
when all fields are changed anyway.

This lint is not applied to structs marked with
[non_exhaustive](https://doc.rust-lang.org/reference/attributes/type_system.html)."##,
    },
    Lint {
        label: "clippy::neg_cmp_op_on_partial_ord",
        description: r##"Checks for the usage of negated comparison operators on types which only implement
`PartialOrd` (e.g., `f64`)."##,
    },
    Lint {
        label: "clippy::neg_multiply",
        description: r##"Checks for multiplication by -1 as a form of negation."##,
    },
    Lint {
        label: "clippy::never_loop",
        description: r##"Checks for loops that will always `break`, `return` or
`continue` an outer loop."##,
    },
    Lint {
        label: "clippy::new_ret_no_self",
        description: r##"Checks for `new` not returning a type that contains `Self`."##,
    },
    Lint {
        label: "clippy::new_without_default",
        description: r##"Checks for types with a `fn new() -> Self` method and no
implementation of
[`Default`](https://doc.rust-lang.org/std/default/trait.Default.html)."##,
    },
    Lint {
        label: "clippy::no_effect",
        description: r##"Checks for statements which have no effect."##,
    },
    Lint {
        label: "clippy::non_ascii_literal",
        description: r##"Checks for non-ASCII characters in string literals."##,
    },
    Lint {
        label: "clippy::non_octal_unix_permissions",
        description: r##"Checks for non-octal values used to set Unix file permissions."##,
    },
    Lint {
        label: "clippy::nonminimal_bool",
        description: r##"Checks for boolean expressions that can be written more
concisely."##,
    },
    Lint {
        label: "clippy::nonsensical_open_options",
        description: r##"Checks for duplicate open options as well as combinations
that make no sense."##,
    },
    Lint {
        label: "clippy::not_unsafe_ptr_arg_deref",
        description: r##"Checks for public functions that dereference raw pointer
arguments but are not marked `unsafe`."##,
    },
    Lint { label: "clippy::ok_expect", description: r##"Checks for usage of `ok().expect(..)`."## },
    Lint {
        label: "clippy::op_ref",
        description: r##"Checks for arguments to `==` which have their address
taken to satisfy a bound
and suggests to dereference the other argument instead"##,
    },
    Lint {
        label: "clippy::option_as_ref_deref",
        description: r##"Checks for usage of `_.as_ref().map(Deref::deref)` or it's aliases (such as String::as_str)."##,
    },
    Lint {
        label: "clippy::option_env_unwrap",
        description: r##"Checks for usage of `option_env!(...).unwrap()` and
suggests usage of the `env!` macro."##,
    },
    Lint {
        label: "clippy::option_filter_map",
        description: r##"Checks for indirect collection of populated `Option`"##,
    },
    Lint {
        label: "clippy::option_if_let_else",
        description: r##"Lints usage of `if let Some(v) = ... { y } else { x }` which is more
idiomatically done with `Option::map_or` (if the else bit is a pure
expression) or `Option::map_or_else` (if the else bit is an impure
expression)."##,
    },
    Lint {
        label: "clippy::option_map_or_none",
        description: r##"Checks for usage of `_.map_or(None, _)`."##,
    },
    Lint {
        label: "clippy::option_map_unit_fn",
        description: r##"Checks for usage of `option.map(f)` where f is a function
or closure that returns the unit type `()`."##,
    },
    Lint {
        label: "clippy::option_option",
        description: r##"Checks for use of `Option<Option<_>>` in function signatures and type
definitions"##,
    },
    Lint {
        label: "clippy::or_fun_call",
        description: r##"Checks for calls to `.or(foo(..))`, `.unwrap_or(foo(..))`,
etc., and suggests to use `or_else`, `unwrap_or_else`, etc., or
`unwrap_or_default` instead."##,
    },
    Lint {
        label: "clippy::out_of_bounds_indexing",
        description: r##"Checks for out of bounds array indexing with a constant
index."##,
    },
    Lint {
        label: "clippy::overflow_check_conditional",
        description: r##"Detects classic underflow/overflow checks."##,
    },
    Lint { label: "clippy::panic", description: r##"Checks for usage of `panic!`."## },
    Lint {
        label: "clippy::panic_in_result_fn",
        description: r##"Checks for usage of `panic!`, `unimplemented!`, `todo!`, `unreachable!` or assertions in a function of type result."##,
    },
    Lint {
        label: "clippy::panicking_unwrap",
        description: r##"Checks for calls of `unwrap[_err]()` that will always fail."##,
    },
    Lint {
        label: "clippy::partialeq_ne_impl",
        description: r##"Checks for manual re-implementations of `PartialEq::ne`."##,
    },
    Lint {
        label: "clippy::path_buf_push_overwrite",
        description: r##"* Checks for [push](https://doc.rust-lang.org/std/path/struct.PathBuf.html#method.push)
calls on `PathBuf` that can cause overwrites."##,
    },
    Lint {
        label: "clippy::pattern_type_mismatch",
        description: r##"Checks for patterns that aren't exact representations of the types
they are applied to.

To satisfy this lint, you will have to adjust either the expression that is matched
against or the pattern itself, as well as the bindings that are introduced by the
adjusted patterns. For matching you will have to either dereference the expression
with the `*` operator, or amend the patterns to explicitly match against `&<pattern>`
or `&mut <pattern>` depending on the reference mutability. For the bindings you need
to use the inverse. You can leave them as plain bindings if you wish for the value
to be copied, but you must use `ref mut <variable>` or `ref <variable>` to construct
a reference into the matched structure.

If you are looking for a way to learn about ownership semantics in more detail, it
is recommended to look at IDE options available to you to highlight types, lifetimes
and reference semantics in your code. The available tooling would expose these things
in a general way even outside of the various pattern matching mechanics. Of course
this lint can still be used to highlight areas of interest and ensure a good understanding
of ownership semantics."##,
    },
    Lint {
        label: "clippy::possible_missing_comma",
        description: r##"Checks for possible missing comma in an array. It lints if
an array element is a binary operator expression and it lies on two lines."##,
    },
    Lint {
        label: "clippy::precedence",
        description: r##"Checks for operations where precedence may be unclear
and suggests to add parentheses. Currently it catches the following:
* mixed usage of arithmetic and bit shifting/combining operators without
parentheses
* a negative numeric literal (which is really a unary `-` followed by a
numeric literal)
  followed by a method call"##,
    },
    Lint {
        label: "clippy::print_literal",
        description: r##"This lint warns about the use of literals as `print!`/`println!` args."##,
    },
    Lint {
        label: "clippy::print_stderr",
        description: r##"Checks for printing on *stderr*. The purpose of this lint
is to catch debugging remnants."##,
    },
    Lint {
        label: "clippy::print_stdout",
        description: r##"Checks for printing on *stdout*. The purpose of this lint
is to catch debugging remnants."##,
    },
    Lint {
        label: "clippy::print_with_newline",
        description: r##"This lint warns when you use `print!()` with a format
string that ends in a newline."##,
    },
    Lint {
        label: "clippy::println_empty_string",
        description: r##"This lint warns when you use `println!()` to
print a newline."##,
    },
    Lint {
        label: "clippy::ptr_arg",
        description: r##"This lint checks for function arguments of type `&String`
or `&Vec` unless the references are mutable. It will also suggest you
replace `.clone()` calls with the appropriate `.to_owned()`/`to_string()`
calls."##,
    },
    Lint {
        label: "clippy::ptr_as_ptr",
        description: r##"Checks for `as` casts between raw pointers without changing its mutability,
namely `*const T` to `*const U` and `*mut T` to `*mut U`."##,
    },
    Lint { label: "clippy::ptr_eq", description: r##"Use `std::ptr::eq` when applicable"## },
    Lint {
        label: "clippy::ptr_offset_with_cast",
        description: r##"Checks for usage of the `offset` pointer method with a `usize` casted to an
`isize`."##,
    },
    Lint {
        label: "clippy::pub_enum_variant_names",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::question_mark",
        description: r##"Checks for expressions that could be replaced by the question mark operator."##,
    },
    Lint {
        label: "clippy::range_minus_one",
        description: r##"Checks for inclusive ranges where 1 is subtracted from
the upper bound, e.g., `x..=(y-1)`."##,
    },
    Lint {
        label: "clippy::range_plus_one",
        description: r##"Checks for exclusive ranges where 1 is added to the
upper bound, e.g., `x..(y+1)`."##,
    },
    Lint {
        label: "clippy::range_step_by_zero",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::range_zip_with_len",
        description: r##"Checks for zipping a collection with the range of
`0.._.len()`."##,
    },
    Lint {
        label: "clippy::rc_buffer",
        description: r##"Checks for `Rc<T>` and `Arc<T>` when `T` is a mutable buffer type such as `String` or `Vec`."##,
    },
    Lint {
        label: "clippy::redundant_allocation",
        description: r##"Checks for use of redundant allocations anywhere in the code."##,
    },
    Lint {
        label: "clippy::redundant_clone",
        description: r##"Checks for a redundant `clone()` (and its relatives) which clones an owned
value that is going to be dropped without further use."##,
    },
    Lint {
        label: "clippy::redundant_closure",
        description: r##"Checks for closures which just call another function where
the function can be called directly. `unsafe` functions or calls where types
get adjusted are ignored."##,
    },
    Lint {
        label: "clippy::redundant_closure_call",
        description: r##"Detects closures called in the same expression where they
are defined."##,
    },
    Lint {
        label: "clippy::redundant_closure_for_method_calls",
        description: r##"Checks for closures which only invoke a method on the closure
argument and can be replaced by referencing the method directly."##,
    },
    Lint {
        label: "clippy::redundant_else",
        description: r##"Checks for `else` blocks that can be removed without changing semantics."##,
    },
    Lint {
        label: "clippy::redundant_field_names",
        description: r##"Checks for fields in struct literals where shorthands
could be used."##,
    },
    Lint {
        label: "clippy::redundant_pattern",
        description: r##"Checks for patterns in the form `name @ _`."##,
    },
    Lint {
        label: "clippy::redundant_pattern_matching",
        description: r##"Lint for redundant pattern matching over `Result`, `Option`,
`std::task::Poll` or `std::net::IpAddr`"##,
    },
    Lint {
        label: "clippy::redundant_pub_crate",
        description: r##"Checks for items declared `pub(crate)` that are not crate visible because they
are inside a private module."##,
    },
    Lint {
        label: "clippy::redundant_slicing",
        description: r##"Checks for redundant slicing expressions which use the full range, and
do not change the type."##,
    },
    Lint {
        label: "clippy::redundant_static_lifetimes",
        description: r##"Checks for constants and statics with an explicit `'static` lifetime."##,
    },
    Lint {
        label: "clippy::ref_binding_to_reference",
        description: r##"Checks for `ref` bindings which create a reference to a reference."##,
    },
    Lint {
        label: "clippy::ref_in_deref",
        description: r##"Checks for references in expressions that use
auto dereference."##,
    },
    Lint {
        label: "clippy::ref_option_ref",
        description: r##"Checks for usage of `&Option<&T>`."##,
    },
    Lint {
        label: "clippy::regex_macro",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::repeat_once",
        description: r##"Checks for usage of `.repeat(1)` and suggest the following method for each types.
- `.to_string()` for `str`
- `.clone()` for `String`
- `.to_vec()` for `slice`"##,
    },
    Lint {
        label: "clippy::replace_consts",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::rest_pat_in_fully_bound_structs",
        description: r##"Checks for unnecessary '..' pattern binding on struct when all fields are explicitly matched."##,
    },
    Lint {
        label: "clippy::result_map_or_into_option",
        description: r##"Checks for usage of `_.map_or(None, Some)`."##,
    },
    Lint {
        label: "clippy::result_map_unit_fn",
        description: r##"Checks for usage of `result.map(f)` where f is a function
or closure that returns the unit type `()`."##,
    },
    Lint {
        label: "clippy::result_unit_err",
        description: r##"Checks for public functions that return a `Result`
with an `Err` type of `()`. It suggests using a custom type that
implements `std::error::Error`."##,
    },
    Lint {
        label: "clippy::reversed_empty_ranges",
        description: r##"Checks for range expressions `x..y` where both `x` and `y`
are constant and `x` is greater or equal to `y`."##,
    },
    Lint {
        label: "clippy::same_functions_in_if_condition",
        description: r##"Checks for consecutive `if`s with the same function call."##,
    },
    Lint {
        label: "clippy::same_item_push",
        description: r##"Checks whether a for loop is being used to push a constant
value into a Vec."##,
    },
    Lint {
        label: "clippy::search_is_some",
        description: r##"Checks for an iterator or string search (such as `find()`,
`position()`, or `rposition()`) followed by a call to `is_some()` or `is_none()`."##,
    },
    Lint {
        label: "clippy::self_assignment",
        description: r##"Checks for explicit self-assignments."##,
    },
    Lint {
        label: "clippy::semicolon_if_nothing_returned",
        description: r##"Looks for blocks of expressions and fires if the last expression returns
`()` but is not followed by a semicolon."##,
    },
    Lint {
        label: "clippy::serde_api_misuse",
        description: r##"Checks for mis-uses of the serde API."##,
    },
    Lint {
        label: "clippy::shadow_reuse",
        description: r##"Checks for bindings that shadow other bindings already in
scope, while reusing the original value."##,
    },
    Lint {
        label: "clippy::shadow_same",
        description: r##"Checks for bindings that shadow other bindings already in
scope, while just changing reference level or mutability."##,
    },
    Lint {
        label: "clippy::shadow_unrelated",
        description: r##"Checks for bindings that shadow other bindings already in
scope, either without a initialization or with one that does not even use
the original value."##,
    },
    Lint {
        label: "clippy::short_circuit_statement",
        description: r##"Checks for the use of short circuit boolean conditions as
a
statement."##,
    },
    Lint {
        label: "clippy::should_assert_eq",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::should_implement_trait",
        description: r##"Checks for methods that should live in a trait
implementation of a `std` trait (see [llogiq's blog
post](http://llogiq.github.io/2015/07/30/traits.html) for further
information) instead of an inherent implementation."##,
    },
    Lint {
        label: "clippy::similar_names",
        description: r##"Checks for names that are very similar and thus confusing."##,
    },
    Lint {
        label: "clippy::single_char_add_str",
        description: r##"Warns when using `push_str`/`insert_str` with a single-character string literal
where `push`/`insert` with a `char` would work fine."##,
    },
    Lint {
        label: "clippy::single_char_pattern",
        description: r##"Checks for string methods that receive a single-character
`str` as an argument, e.g., `_.split(x)`."##,
    },
    Lint {
        label: "clippy::single_component_path_imports",
        description: r##"Checking for imports with single component use path."##,
    },
    Lint {
        label: "clippy::single_element_loop",
        description: r##"Checks whether a for loop has a single element."##,
    },
    Lint {
        label: "clippy::single_match",
        description: r##"Checks for matches with a single arm where an `if let`
will usually suffice."##,
    },
    Lint {
        label: "clippy::single_match_else",
        description: r##"Checks for matches with two arms where an `if let else` will
usually suffice."##,
    },
    Lint {
        label: "clippy::size_of_in_element_count",
        description: r##"Detects expressions where
`size_of::<T>` or `size_of_val::<T>` is used as a
count of elements of type `T`"##,
    },
    Lint {
        label: "clippy::skip_while_next",
        description: r##"Checks for usage of `_.skip_while(condition).next()`."##,
    },
    Lint {
        label: "clippy::slow_vector_initialization",
        description: r##"Checks slow zero-filled vector initialization"##,
    },
    Lint {
        label: "clippy::stable_sort_primitive",
        description: r##"When sorting primitive values (integers, bools, chars, as well
as arrays, slices, and tuples of such items), it is better to
use an unstable sort than a stable sort."##,
    },
    Lint {
        label: "clippy::str_to_string",
        description: r##"This lint checks for `.to_string()` method calls on values of type `&str`."##,
    },
    Lint {
        label: "clippy::string_add",
        description: r##"Checks for all instances of `x + _` where `x` is of type
`String`, but only if [`string_add_assign`](#string_add_assign) does *not*
match."##,
    },
    Lint {
        label: "clippy::string_add_assign",
        description: r##"Checks for string appends of the form `x = x + y` (without
`let`!)."##,
    },
    Lint {
        label: "clippy::string_extend_chars",
        description: r##"Checks for the use of `.extend(s.chars())` where s is a
`&str` or `String`."##,
    },
    Lint {
        label: "clippy::string_from_utf8_as_bytes",
        description: r##"Check if the string is transformed to byte array and casted back to string."##,
    },
    Lint {
        label: "clippy::string_lit_as_bytes",
        description: r##"Checks for the `as_bytes` method called on string literals
that contain only ASCII characters."##,
    },
    Lint {
        label: "clippy::string_to_string",
        description: r##"This lint checks for `.to_string()` method calls on values of type `String`."##,
    },
    Lint {
        label: "clippy::struct_excessive_bools",
        description: r##"Checks for excessive
use of bools in structs."##,
    },
    Lint {
        label: "clippy::suboptimal_flops",
        description: r##"Looks for floating-point expressions that
can be expressed using built-in methods to improve both
accuracy and performance."##,
    },
    Lint {
        label: "clippy::suspicious_arithmetic_impl",
        description: r##"Lints for suspicious operations in impls of arithmetic operators, e.g.
subtracting elements in an Add impl."##,
    },
    Lint {
        label: "clippy::suspicious_assignment_formatting",
        description: r##"Checks for use of the non-existent `=*`, `=!` and `=-`
operators."##,
    },
    Lint {
        label: "clippy::suspicious_else_formatting",
        description: r##"Checks for formatting of `else`. It lints if the `else`
is followed immediately by a newline or the `else` seems to be missing."##,
    },
    Lint {
        label: "clippy::suspicious_map",
        description: r##"Checks for calls to `map` followed by a `count`."##,
    },
    Lint {
        label: "clippy::suspicious_op_assign_impl",
        description: r##"Lints for suspicious operations in impls of OpAssign, e.g.
subtracting elements in an AddAssign impl."##,
    },
    Lint {
        label: "clippy::suspicious_operation_groupings",
        description: r##"Checks for unlikely usages of binary operators that are almost
certainly typos and/or copy/paste errors, given the other usages
of binary operators nearby."##,
    },
    Lint {
        label: "clippy::suspicious_splitn",
        description: r##"Checks for calls to [`splitn`]
(https://doc.rust-lang.org/std/primitive.str.html#method.splitn) and
related functions with either zero or one splits."##,
    },
    Lint {
        label: "clippy::suspicious_unary_op_formatting",
        description: r##"Checks the formatting of a unary operator on the right hand side
of a binary operator. It lints if there is no space between the binary and unary operators,
but there is a space between the unary and its operand."##,
    },
    Lint {
        label: "clippy::tabs_in_doc_comments",
        description: r##"Checks doc comments for usage of tab characters."##,
    },
    Lint {
        label: "clippy::temporary_assignment",
        description: r##"Checks for construction of a structure or tuple just to
assign a value in it."##,
    },
    Lint {
        label: "clippy::to_digit_is_some",
        description: r##"Checks for `.to_digit(..).is_some()` on `char`s."##,
    },
    Lint {
        label: "clippy::to_string_in_display",
        description: r##"Checks for uses of `to_string()` in `Display` traits."##,
    },
    Lint { label: "clippy::todo", description: r##"Checks for usage of `todo!`."## },
    Lint {
        label: "clippy::too_many_arguments",
        description: r##"Checks for functions with too many parameters."##,
    },
    Lint {
        label: "clippy::too_many_lines",
        description: r##"Checks for functions with a large amount of lines."##,
    },
    Lint {
        label: "clippy::toplevel_ref_arg",
        description: r##"Checks for function arguments and let bindings denoted as
`ref`."##,
    },
    Lint {
        label: "clippy::trait_duplication_in_bounds",
        description: r##"Checks for cases where generics are being used and multiple
syntax specifications for trait bounds are used simultaneously."##,
    },
    Lint {
        label: "clippy::transmute_bytes_to_str",
        description: r##"Checks for transmutes from a `&[u8]` to a `&str`."##,
    },
    Lint {
        label: "clippy::transmute_float_to_int",
        description: r##"Checks for transmutes from a float to an integer."##,
    },
    Lint {
        label: "clippy::transmute_int_to_bool",
        description: r##"Checks for transmutes from an integer to a `bool`."##,
    },
    Lint {
        label: "clippy::transmute_int_to_char",
        description: r##"Checks for transmutes from an integer to a `char`."##,
    },
    Lint {
        label: "clippy::transmute_int_to_float",
        description: r##"Checks for transmutes from an integer to a float."##,
    },
    Lint {
        label: "clippy::transmute_ptr_to_ptr",
        description: r##"Checks for transmutes from a pointer to a pointer, or
from a reference to a reference."##,
    },
    Lint {
        label: "clippy::transmute_ptr_to_ref",
        description: r##"Checks for transmutes from a pointer to a reference."##,
    },
    Lint {
        label: "clippy::transmutes_expressible_as_ptr_casts",
        description: r##"Checks for transmutes that could be a pointer cast."##,
    },
    Lint {
        label: "clippy::transmuting_null",
        description: r##"Checks for transmute calls which would receive a null pointer."##,
    },
    Lint {
        label: "clippy::trivial_regex",
        description: r##"Checks for trivial [regex](https://crates.io/crates/regex)
creation (with `Regex::new`, `RegexBuilder::new`, or `RegexSet::new`)."##,
    },
    Lint {
        label: "clippy::trivially_copy_pass_by_ref",
        description: r##"Checks for functions taking arguments by reference, where
the argument type is `Copy` and small enough to be more efficient to always
pass by value."##,
    },
    Lint { label: "clippy::try_err", description: r##"Checks for usages of `Err(x)?`."## },
    Lint {
        label: "clippy::type_complexity",
        description: r##"Checks for types used in structs, parameters and `let`
declarations above a certain complexity threshold."##,
    },
    Lint {
        label: "clippy::type_repetition_in_bounds",
        description: r##"This lint warns about unnecessary type repetitions in trait bounds"##,
    },
    Lint {
        label: "clippy::undropped_manually_drops",
        description: r##"Prevents the safe `std::mem::drop` function from being called on `std::mem::ManuallyDrop`."##,
    },
    Lint {
        label: "clippy::unicode_not_nfc",
        description: r##"Checks for string literals that contain Unicode in a form
that is not equal to its
[NFC-recomposition](http://www.unicode.org/reports/tr15/#Norm_Forms)."##,
    },
    Lint {
        label: "clippy::unimplemented",
        description: r##"Checks for usage of `unimplemented!`."##,
    },
    Lint {
        label: "clippy::uninit_assumed_init",
        description: r##"Checks for `MaybeUninit::uninit().assume_init()`."##,
    },
    Lint {
        label: "clippy::unit_arg",
        description: r##"Checks for passing a unit value as an argument to a function without using a
unit literal (`()`)."##,
    },
    Lint {
        label: "clippy::unit_cmp",
        description: r##"Checks for comparisons to unit. This includes all binary
comparisons (like `==` and `<`) and asserts."##,
    },
    Lint {
        label: "clippy::unit_return_expecting_ord",
        description: r##"Checks for functions that expect closures of type
Fn(...) -> Ord where the implemented closure returns the unit type.
The lint also suggests to remove the semi-colon at the end of the statement if present."##,
    },
    Lint {
        label: "clippy::unnecessary_cast",
        description: r##"Checks for casts to the same type, casts of int literals to integer types
and casts of float literals to float types."##,
    },
    Lint {
        label: "clippy::unnecessary_filter_map",
        description: r##"Checks for `filter_map` calls which could be replaced by `filter` or `map`.
More specifically it checks if the closure provided is only performing one of the
filter or map operations and suggests the appropriate option."##,
    },
    Lint {
        label: "clippy::unnecessary_fold",
        description: r##"Checks for using `fold` when a more succinct alternative exists.
Specifically, this checks for `fold`s which could be replaced by `any`, `all`,
`sum` or `product`."##,
    },
    Lint {
        label: "clippy::unnecessary_lazy_evaluations",
        description: r##"As the counterpart to `or_fun_call`, this lint looks for unnecessary
lazily evaluated closures on `Option` and `Result`.

This lint suggests changing the following functions, when eager evaluation results in
simpler code:
 - `unwrap_or_else` to `unwrap_or`
 - `and_then` to `and`
 - `or_else` to `or`
 - `get_or_insert_with` to `get_or_insert`
 - `ok_or_else` to `ok_or`"##,
    },
    Lint {
        label: "clippy::unnecessary_mut_passed",
        description: r##"Detects passing a mutable reference to a function that only
requires an immutable reference."##,
    },
    Lint {
        label: "clippy::unnecessary_operation",
        description: r##"Checks for expression statements that can be reduced to a
sub-expression."##,
    },
    Lint {
        label: "clippy::unnecessary_self_imports",
        description: r##"Checks for imports ending in `::{self}`."##,
    },
    Lint {
        label: "clippy::unnecessary_sort_by",
        description: r##"Detects uses of `Vec::sort_by` passing in a closure
which compares the two arguments, either directly or indirectly."##,
    },
    Lint {
        label: "clippy::unnecessary_unwrap",
        description: r##"Checks for calls of `unwrap[_err]()` that cannot fail."##,
    },
    Lint {
        label: "clippy::unnecessary_wraps",
        description: r##"Checks for private functions that only return `Ok` or `Some`."##,
    },
    Lint {
        label: "clippy::unneeded_field_pattern",
        description: r##"Checks for structure field patterns bound to wildcards."##,
    },
    Lint {
        label: "clippy::unneeded_wildcard_pattern",
        description: r##"Checks for tuple patterns with a wildcard
pattern (`_`) is next to a rest pattern (`..`).

_NOTE_: While `_, ..` means there is at least one element left, `..`
means there are 0 or more elements left. This can make a difference
when refactoring, but shouldn't result in errors in the refactored code,
since the wildcard pattern isn't used anyway."##,
    },
    Lint {
        label: "clippy::unnested_or_patterns",
        description: r##"Checks for unnested or-patterns, e.g., `Some(0) | Some(2)` and
suggests replacing the pattern with a nested one, `Some(0 | 2)`.

Another way to think of this is that it rewrites patterns in
*disjunctive normal form (DNF)* into *conjunctive normal form (CNF)*."##,
    },
    Lint { label: "clippy::unreachable", description: r##"Checks for usage of `unreachable!`."## },
    Lint {
        label: "clippy::unreadable_literal",
        description: r##"Warns if a long integral or floating-point constant does
not contain underscores."##,
    },
    Lint {
        label: "clippy::unsafe_derive_deserialize",
        description: r##"Checks for deriving `serde::Deserialize` on a type that
has methods using `unsafe`."##,
    },
    Lint {
        label: "clippy::unsafe_removed_from_name",
        description: r##"Checks for imports that remove unsafe from an item's
name."##,
    },
    Lint {
        label: "clippy::unsafe_vector_initialization",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::unseparated_literal_suffix",
        description: r##"Warns if literal suffixes are not separated by an
underscore."##,
    },
    Lint {
        label: "clippy::unsound_collection_transmute",
        description: r##"Checks for transmutes between collections whose
types have different ABI, size or alignment."##,
    },
    Lint {
        label: "clippy::unstable_as_mut_slice",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::unstable_as_slice",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::unused_async",
        description: r##"Checks for functions that are declared `async` but have no `.await`s inside of them."##,
    },
    Lint {
        label: "clippy::unused_collect",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::unused_io_amount",
        description: r##"Checks for unused written/read amount."##,
    },
    Lint {
        label: "clippy::unused_self",
        description: r##"Checks methods that contain a `self` argument but don't use it"##,
    },
    Lint {
        label: "clippy::unused_unit",
        description: r##"Checks for unit (`()`) expressions that can be removed."##,
    },
    Lint {
        label: "clippy::unusual_byte_groupings",
        description: r##"Warns if hexadecimal or binary literals are not grouped
by nibble or byte."##,
    },
    Lint {
        label: "clippy::unwrap_in_result",
        description: r##"Checks for functions of type Result that contain `expect()` or `unwrap()`"##,
    },
    Lint {
        label: "clippy::unwrap_used",
        description: r##"Checks for `.unwrap()` calls on `Option`s and on `Result`s."##,
    },
    Lint {
        label: "clippy::upper_case_acronyms",
        description: r##"Checks for fully capitalized names and optionally names containing a capitalized acronym."##,
    },
    Lint {
        label: "clippy::use_debug",
        description: r##"Checks for use of `Debug` formatting. The purpose of this
lint is to catch debugging remnants."##,
    },
    Lint {
        label: "clippy::use_self",
        description: r##"Checks for unnecessary repetition of structure name when a
replacement with `Self` is applicable."##,
    },
    Lint {
        label: "clippy::used_underscore_binding",
        description: r##"Checks for the use of bindings with a single leading
underscore."##,
    },
    Lint {
        label: "clippy::useless_asref",
        description: r##"Checks for usage of `.as_ref()` or `.as_mut()` where the
types before and after the call are the same."##,
    },
    Lint {
        label: "clippy::useless_attribute",
        description: r##"Checks for `extern crate` and `use` items annotated with
lint attributes.

This lint permits `#[allow(unused_imports)]`, `#[allow(deprecated)]`,
`#[allow(unreachable_pub)]`, `#[allow(clippy::wildcard_imports)]` and
`#[allow(clippy::enum_glob_use)]` on `use` items and `#[allow(unused_imports)]` on
`extern crate` items with a `#[macro_use]` attribute."##,
    },
    Lint {
        label: "clippy::useless_conversion",
        description: r##"Checks for `Into`, `TryInto`, `From`, `TryFrom`, or `IntoIter` calls
which uselessly convert to the same type."##,
    },
    Lint {
        label: "clippy::useless_format",
        description: r##"Checks for the use of `format!(string literal with no
argument)` and `format!({}, foo)` where `foo` is a string."##,
    },
    Lint {
        label: "clippy::useless_let_if_seq",
        description: r##"Checks for variable declarations immediately followed by a
conditional affectation."##,
    },
    Lint {
        label: "clippy::useless_transmute",
        description: r##"Checks for transmutes to the original type of the object
and transmutes that could be a cast."##,
    },
    Lint {
        label: "clippy::useless_vec",
        description: r##"Checks for usage of `&vec![..]` when using `&[..]` would
be possible."##,
    },
    Lint {
        label: "clippy::vec_box",
        description: r##"Checks for use of `Vec<Box<T>>` where T: Sized anywhere in the code.
Check the [Box documentation](https://doc.rust-lang.org/std/boxed/index.html) for more information."##,
    },
    Lint {
        label: "clippy::vec_init_then_push",
        description: r##"Checks for calls to `push` immediately after creating a new `Vec`."##,
    },
    Lint {
        label: "clippy::vec_resize_to_zero",
        description: r##"Finds occurrences of `Vec::resize(0, an_int)`"##,
    },
    Lint {
        label: "clippy::verbose_bit_mask",
        description: r##"Checks for bit masks that can be replaced by a call
to `trailing_zeros`"##,
    },
    Lint {
        label: "clippy::verbose_file_reads",
        description: r##"Checks for use of File::read_to_end and File::read_to_string."##,
    },
    Lint {
        label: "clippy::vtable_address_comparisons",
        description: r##"Checks for comparisons with an address of a trait vtable."##,
    },
    Lint {
        label: "clippy::while_immutable_condition",
        description: r##"Checks whether variables used within while loop condition
can be (and are) mutated in the body."##,
    },
    Lint {
        label: "clippy::while_let_loop",
        description: r##"Detects `loop + match` combinations that are easier
written as a `while let` loop."##,
    },
    Lint {
        label: "clippy::while_let_on_iterator",
        description: r##"Checks for `while let` expressions on iterators."##,
    },
    Lint {
        label: "clippy::wildcard_dependencies",
        description: r##"Checks for wildcard dependencies in the `Cargo.toml`."##,
    },
    Lint {
        label: "clippy::wildcard_enum_match_arm",
        description: r##"Checks for wildcard enum matches using `_`."##,
    },
    Lint {
        label: "clippy::wildcard_imports",
        description: r##"Checks for wildcard imports `use _::*`."##,
    },
    Lint {
        label: "clippy::wildcard_in_or_patterns",
        description: r##"Checks for wildcard pattern used with others patterns in same match arm."##,
    },
    Lint {
        label: "clippy::write_literal",
        description: r##"This lint warns about the use of literals as `write!`/`writeln!` args."##,
    },
    Lint {
        label: "clippy::write_with_newline",
        description: r##"This lint warns when you use `write!()` with a format
string that
ends in a newline."##,
    },
    Lint {
        label: "clippy::writeln_empty_string",
        description: r##"This lint warns when you use `writeln!(buf, )` to
print a newline."##,
    },
    Lint {
        label: "clippy::wrong_pub_self_convention",
        description: r##"Nothing. This lint has been deprecated."##,
    },
    Lint {
        label: "clippy::wrong_self_convention",
        description: r##"Checks for methods with certain name prefixes and which
doesn't match how self is taken. The actual rules are:

|Prefix |Postfix     |`self` taken           | `self` type  |
|-------|------------|-----------------------|--------------|
|`as_`  | none       |`&self` or `&mut self` | any          |
|`from_`| none       | none                  | any          |
|`into_`| none       |`self`                 | any          |
|`is_`  | none       |`&self` or none        | any          |
|`to_`  | `_mut`     |`&mut self`            | any          |
|`to_`  | not `_mut` |`self`                 | `Copy`       |
|`to_`  | not `_mut` |`&self`                | not `Copy`   |

Note: Clippy doesn't trigger methods with `to_` prefix in:
- Traits definition.
Clippy can not tell if a type that implements a trait is `Copy` or not.
- Traits implementation, when `&self` is taken.
The method signature is controlled by the trait and often `&self` is required for all types that implement the trait
(see e.g. the `std::string::ToString` trait).

Please find more info here:
https://rust-lang.github.io/api-guidelines/naming.html#ad-hoc-conversions-follow-as_-to_-into_-conventions-c-conv"##,
    },
    Lint {
        label: "clippy::wrong_transmute",
        description: r##"Checks for transmutes that can't ever be correct on any
architecture."##,
    },
    Lint { label: "clippy::zero_divided_by_zero", description: r##"Checks for `0.0 / 0.0`."## },
    Lint {
        label: "clippy::zero_prefixed_literal",
        description: r##"Warns if an integral constant literal starts with `0`."##,
    },
    Lint {
        label: "clippy::zero_ptr",
        description: r##"Catch casts from `0` to some pointer type"##,
    },
    Lint {
        label: "clippy::zero_sized_map_values",
        description: r##"Checks for maps with zero-sized value types anywhere in the code."##,
    },
    Lint {
        label: "clippy::zst_offset",
        description: r##"Checks for `offset(_)`, `wrapping_`{`add`, `sub`}, etc. on raw pointers to
zero-sized types"##,
    },
];
