typeck-field-multiply-specified-in-initializer =
    field `{$ident}` specified more than once
    .label = used more than once
    .previous-use-label = first use of `{$ident}`

typeck-unrecognized-atomic-operation =
    unrecognized atomic operation function: `{$op}`
    .label = unrecognized atomic operation

typeck-wrong-number-of-generic-arguments-to-intrinsic =
    intrinsic has wrong number of {$descr} parameters: found {$found}, expected {$expected}
    .label = expected {$expected} {$descr} {$expected ->
        [one] parameter
        *[other] parameters
    }

typeck-unrecognized-intrinsic-function =
    unrecognized intrinsic function: `{$name}`
    .label = unrecognized intrinsic

typeck-lifetimes-or-bounds-mismatch-on-trait =
    lifetime parameters or bounds on {$item_kind} `{$ident}` do not match the trait declaration
    .label = lifetimes do not match {$item_kind} in trait
    .generics-label = lifetimes in impl do not match this {$item_kind} in trait

typeck-drop-impl-on-wrong-item =
    the `Drop` trait may only be implemented for structs, enums, and unions
    .label = must be a struct, enum, or union

typeck-field-already-declared =
    field `{$field_name}` is already declared
    .label = field already declared
    .previous-decl-label = `{$field_name}` first declared here

typeck-copy-impl-on-type-with-dtor =
    the trait `Copy` may not be implemented for this type; the type has a destructor
    .label = `Copy` not allowed on types with destructors

typeck-multiple-relaxed-default-bounds =
    type parameter has more than one relaxed default bound, only one is supported

typeck-copy-impl-on-non-adt =
    the trait `Copy` may not be implemented for this type
    .label = type is not a structure or enumeration

typeck-trait-object-declared-with-no-traits =
    at least one trait is required for an object type

typeck-ambiguous-lifetime-bound =
    ambiguous lifetime bound, explicit lifetime bound required

typeck-assoc-type-binding-not-allowed =
    associated type bindings are not allowed here
    .label = associated type not allowed here

typeck-functional-record-update-on-non-struct =
    functional record update syntax requires a struct

typeck-typeof-reserved-keyword-used =
    `typeof` is a reserved keyword but unimplemented
    .suggestion = consider replacing `typeof(...)` with an actual type
    .label = reserved keyword

typeck-return-stmt-outside-of-fn-body =
    return statement outside of function body
    .encl-body-label = the return is part of this body...
    .encl-fn-label = ...not the enclosing function body

typeck-yield-expr-outside-of-generator =
    yield expression outside of generator literal

typeck-struct-expr-non-exhaustive =
    cannot create non-exhaustive {$what} using struct expression

typeck-method-call-on-unknown-type =
    the type of this value must be known to call a method on a raw pointer on it

typeck-value-of-associated-struct-already-specified =
    the value of the associated type `{$item_name}` (from trait `{$def_path}`) is already specified
    .label = re-bound here
    .previous-bound-label = `{$item_name}` bound here first

typeck-address-of-temporary-taken = cannot take address of a temporary
    .label = temporary value

typeck-add-return-type-add = try adding a return type

typeck-add-return-type-missing-here = a return type might be missing here

typeck-expected-default-return-type = expected `()` because of default return type

typeck-expected-return-type = expected `{$expected}` because of return type

typeck-unconstrained-opaque-type = unconstrained opaque type
    .note = `{$name}` must be used in combination with a concrete type within the same module

typeck-explicit-generic-args-with-impl-trait =
    cannot provide explicit generic arguments when `impl Trait` is used in argument position
    .label = explicit generic argument not allowed
    .note = see issue #83701 <https://github.com/rust-lang/rust/issues/83701> for more information

typeck-explicit-generic-args-with-impl-trait-feature =
    add `#![feature(explicit_generic_args_with_impl_trait)]` to the crate attributes to enable
