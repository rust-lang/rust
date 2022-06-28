lint-array-into-iter =
    this method call resolves to `<&{$target} as IntoIterator>::into_iter` (due to backwards compatibility), but will resolve to <{$target} as IntoIterator>::into_iter in Rust 2021
    .use-iter-suggestion = use `.iter()` instead of `.into_iter()` to avoid ambiguity
    .remove-into-iter-suggestion = or remove `.into_iter()` to iterate by value
    .use-explicit-into-iter-suggestion =
        or use `IntoIterator::into_iter(..)` instead of `.into_iter()` to explicitly iterate by value

lint-enum-intrinsics-mem-discriminant =
    the return value of `mem::discriminant` is unspecified when called with a non-enum type
    .note = the argument to `discriminant` should be a reference to an enum, but it was passed a reference to a `{$ty_param}`, which is not an enum.

lint-enum-intrinsics-mem-variant =
    the return value of `mem::variant_count` is unspecified when called with a non-enum type
    .note = the type parameter of `variant_count` should be an enum, but it was instantiated with the type `{$ty_param}`, which is not an enum.

lint-expectation = this lint expectation is unfulfilled
    .note = the `unfulfilled_lint_expectations` lint can't be expected and will always produce this message

lint-hidden-unicode-codepoints = unicode codepoint changing visible direction of text present in {$label}
    .label = this {$label} contains {$count ->
        [one] an invisible
        *[other] invisible
    } unicode text flow control {$count ->
        [one] codepoint
        *[other] codepoints
    }
    .note = these kind of unicode codepoints change the way text flows on applications that support them, but can cause confusion because they change the order of characters on the screen
    .suggestion-remove = if their presence wasn't intentional, you can remove them
    .suggestion-escape = if you want to keep them but make them visible in your source code, you can escape them
    .no-suggestion-note-escape = if you want to keep them but make them visible in your source code, you can escape them: {$escaped}

lint-default-hash-types = prefer `{$preferred}` over `{$used}`, it has better performance
    .note = a `use rustc_data_structures::fx::{$preferred}` may be necessary

lint-query-instability = using `{$query}` can result in unstable query results
    .note = if you believe this case to be fine, allow this lint and add a comment explaining your rationale

lint-tykind-kind = usage of `ty::TyKind::<kind>`
    .suggestion = try using `ty::<kind>` directly

lint-tykind = usage of `ty::TyKind`
    .help = try using `Ty` instead

lint-ty-qualified = usage of qualified `ty::{$ty}`
    .suggestion = try importing it and using it unqualified

lint-lintpass-by-hand = implementing `LintPass` by hand
    .help = try using `declare_lint_pass!` or `impl_lint_pass!` instead

lint-non-existant-doc-keyword = found non-existing keyword `{$keyword}` used in `#[doc(keyword = \"...\")]`
    .help = only existing keywords are allowed in core/std

lint-diag-out-of-impl =
    diagnostics should only be created in `SessionDiagnostic`/`AddSubdiagnostic` impls

lint-untranslatable-diag = diagnostics should be created using translatable messages

lint-cstring-ptr = getting the inner pointer of a temporary `CString`
    .as-ptr-label = this pointer will be invalid
    .unwrap-label = this `CString` is deallocated at the end of the statement, bind it to a variable to extend its lifetime
    .note = pointers do not have a lifetime; when calling `as_ptr` the `CString` will be deallocated at the end of the statement because nothing is referencing it as far as the type system is concerned
    .help = for more information, see https://doc.rust-lang.org/reference/destructors.html

lint-identifier-non-ascii-char = identifier contains non-ASCII characters

lint-identifier-uncommon-codepoints = identifier contains uncommon Unicode codepoints

lint-confusable-identifier-pair = identifier pair considered confusable between `{$existing_sym}` and `{$sym}`
    .label = this is where the previous identifier occurred

lint-mixed-script-confusables =
    the usage of Script Group `{$set}` in this crate consists solely of mixed script confusables
    .includes-note = the usage includes {$includes}
    .note = please recheck to make sure their usages are indeed what you want

lint-non-fmt-panic = panic message is not a string literal
    .note = this usage of `{$name}!()` is deprecated; it will be a hard error in Rust 2021
    .more-info-note = for more information, see <https://doc.rust-lang.org/nightly/edition-guide/rust-2021/panic-macro-consistency.html>
    .supports-fmt-note = the `{$name}!()` macro supports formatting, so there's no need for the `format!()` macro here
    .supports-fmt-suggestion = remove the `format!(..)` macro call
    .display-suggestion = add a "{"{"}{"}"}" format string to `Display` the message
    .debug-suggestion =
        add a "{"{"}:?{"}"}" format string to use the `Debug` implementation of `{$ty}`
    .panic-suggestion = {$already_suggested ->
        [true] or use
        *[false] use
    } std::panic::panic_any instead

lint-non-fmt-panic-unused =
    panic message contains {$count ->
        [one] an unused
        *[other] unused
    } formatting {$count ->
        [one] placeholder
        *[other] placeholders
    }
    .note = this message is not used as a format string when given without arguments, but will be in Rust 2021
    .add-args-suggestion = add the missing {$count ->
        [one] argument
        *[other] arguments
    }
    .add-fmt-suggestion = or add a "{"{"}{"}"}" format string to use the message literally

lint-non-fmt-panic-braces =
    panic message contains {$count ->
        [one] a brace
        *[other] braces
    }
    .note = this message is not used as a format string, but will be in Rust 2021
    .suggestion = add a "{"{"}{"}"}" format string to use the message literally

lint-non-camel-case-type = {$sort} `{$name}` should have an upper camel case name
    .suggestion = convert the identifier to upper camel case
    .label = should have an UpperCamelCase name

lint-non-snake-case = {$sort} `{$name}` should have a snake case name
    .rename-or-convert-suggestion = rename the identifier or convert it to a snake case raw identifier
    .cannot-convert-note = `{$sc}` cannot be used as a raw identifier
    .rename-suggestion = rename the identifier
    .convert-suggestion = convert the identifier to snake case
    .help = convert the identifier to snake case: `{$sc}`
    .label = should have a snake_case name

lint-non-upper_case-global = {$sort} `{$name}` should have an upper case name
    .suggestion = convert the identifier to upper case
    .label = should have an UPPER_CASE name

lint-noop-method-call = call to `.{$method}()` on a reference in this situation does nothing
    .label = unnecessary method call
    .note = the type `{$receiver_ty}` which `{$method}` is being called on is the same as the type returned from `{$method}`, so the method call does not do anything and can be removed

lint-pass-by-value = passing `{$ty}` by reference
    .suggestion = try passing by value

lint-redundant-semicolons =
    unnecessary trailing {$multiple ->
        [true] semicolons
        *[false] semicolon
    }
    .suggestion = remove {$multiple ->
        [true] these semicolons
        *[false] this semicolon
    }

lint-drop-trait-constraints =
    bounds on `{$predicate}` are most likely incorrect, consider instead using `{$needs_drop}` to detect whether a type can be trivially dropped

lint-drop-glue =
    types that do not implement `Drop` can still have drop glue, consider instead using `{$needs_drop}` to detect whether a type is trivially dropped

lint-range-endpoint-out-of-range = range endpoint is out of range for `{$ty}`
    .suggestion = use an inclusive range instead

lint-overflowing-bin-hex = literal out of range for `{$ty}`
    .negative-note = the literal `{$lit}` (decimal `{$dec}`) does not fit into the type `{$ty}`
    .negative-becomes-note = and the value `-{$lit}` will become `{$actually}{$ty}`
    .positive-note = the literal `{$lit}` (decimal `{$dec}`) does not fit into the type `{$ty}` and will become `{$actually}{$ty}`
    .suggestion = consider using the type `{$suggestion_ty}` instead
    .help = consider using the type `{$suggestion_ty}` instead

lint-overflowing-int = literal out of range for `{$ty}`
    .note = the literal `{$lit}` does not fit into the type `{$ty}` whose range is `{$min}..={$max}`
    .help = consider using the type `{$suggestion_ty}` instead

lint-only-cast-u8-to-char = only `u8` can be cast into `char`
    .suggestion = use a `char` literal instead

lint-overflowing-uint = literal out of range for `{$ty}`
    .note = the literal `{$lit}` does not fit into the type `{$ty}` whose range is `{$min}..={$max}`

lint-overflowing-literal = literal out of range for `{$ty}`
    .note = the literal `{$lit}` does not fit into the type `{$ty}` and will be converted to `{$ty}::INFINITY`

lint-unused-comparisons = comparison is useless due to type limits

lint-improper-ctypes = `extern` {$desc} uses type `{$ty}`, which is not FFI-safe
    .label = not FFI-safe
    .note = the type is defined here

lint-improper-ctypes-opaque = opaque types have no C equivalent

lint-improper-ctypes-fnptr-reason = this function pointer has Rust-specific calling convention
lint-improper-ctypes-fnptr-help = consider using an `extern fn(...) -> ...` function pointer instead

lint-improper-ctypes-tuple-reason = tuples have unspecified layout
lint-improper-ctypes-tuple-help = consider using a struct instead

lint-improper-ctypes-str-reason = string slices have no C equivalent
lint-improper-ctypes-str-help = consider using `*const u8` and a length instead

lint-improper-ctypes-dyn = trait objects have no C equivalent

lint-improper-ctypes-slice-reason = slices have no C equivalent
lint-improper-ctypes-slice-help = consider using a raw pointer instead

lint-improper-ctypes-128bit = 128-bit integers don't currently have a known stable ABI

lint-improper-ctypes-char-reason = the `char` type has no C equivalent
lint-improper-ctypes-char-help = consider using `u32` or `libc::wchar_t` instead

lint-improper-ctypes-non-exhaustive = this enum is non-exhaustive
lint-improper-ctypes-non-exhaustive-variant = this enum has non-exhaustive variants

lint-improper-ctypes-enum-repr-reason = enum has no representation hint
lint-improper-ctypes-enum-repr-help =
    consider adding a `#[repr(C)]`, `#[repr(transparent)]`, or integer `#[repr(...)]` attribute to this enum

lint-improper-ctypes-struct-fieldless-reason = this struct has no fields
lint-improper-ctypes-struct-fieldless-help = consider adding a member to this struct

lint-improper-ctypes-union-fieldless-reason = this union has no fields
lint-improper-ctypes-union-fieldless-help = consider adding a member to this union

lint-improper-ctypes-struct-non-exhaustive = this struct is non-exhaustive
lint-improper-ctypes-union-non-exhaustive = this union is non-exhaustive

lint-improper-ctypes-struct-layout-reason = this struct has unspecified layout
lint-improper-ctypes-struct-layout-help = consider adding a `#[repr(C)]` or `#[repr(transparent)]` attribute to this struct

lint-improper-ctypes-union-layout-reason = this union has unspecified layout
lint-improper-ctypes-union-layout-help = consider adding a `#[repr(C)]` or `#[repr(transparent)]` attribute to this union

lint-improper-ctypes-box = box cannot be represented as a single pointer

lint-improper-ctypes-enum-phantomdata = this enum contains a PhantomData field

lint-improper-ctypes-struct-zst = this struct contains only zero-sized fields

lint-improper-ctypes-array-reason = passing raw arrays by value is not FFI-safe
lint-improper-ctypes-array-help = consider passing a pointer to the array

lint-improper-ctypes-only-phantomdata = composed only of `PhantomData`

lint-variant-size-differences =
    enum variant is more than three times larger ({$largest} bytes) than the next largest

lint-atomic-ordering-load = atomic loads cannot have `Release` or `AcqRel` ordering
    .help = consider using ordering modes `Acquire`, `SeqCst` or `Relaxed`

lint-atomic-ordering-store = atomic stores cannot have `Acquire` or `AcqRel` ordering
    .help = consider using ordering modes `Release`, `SeqCst` or `Relaxed`

lint-atomic-ordering-fence = memory fences cannot have `Relaxed` ordering
    .help = consider using ordering modes `Acquire`, `Release`, `AcqRel` or `SeqCst`

lint-atomic-ordering-invalid = `{$method}`'s failure ordering may not be `Release` or `AcqRel`, since a failed `{$method}` does not result in a write
    .label = invalid failure ordering
    .help = consider using `Acquire` or `Relaxed` failure ordering instead

lint-atomic-ordering-invalid-fail-success = `{$method}`'s success ordering must be at least as strong as its failure ordering
    .fail-label = `{$fail_ordering}` failure ordering
    .success-label = `{$success_ordering}` success ordering
    .suggestion = consider using `{$success_suggestion}` success ordering instead

lint-unused-op = unused {$op} that must be used
    .label = the {$op} produces a value
    .suggestion = use `let _ = ...` to ignore the resulting value

lint-unused-result = unused result of type `{$ty}`

lint-unused-closure =
    unused {$pre}{$count ->
        [one] closure
        *[other] closures
    }{$post} that must be used
    .note = closures are lazy and do nothing unless called

lint-unused-generator =
    unused {$pre}{$count ->
        [one] generator
        *[other] generator
    }{$post} that must be used
    .note = generators are lazy and do nothing unless resumed

lint-unused-def = unused {$pre}`{$def}`{$post} that must be used

lint-path-statement-drop = path statement drops value
    .suggestion = use `drop` to clarify the intent

lint-path-statement-no-effect = path statement with no effect

lint-unused-delim = unnecessary {$delim} around {$item}
    .suggestion = remove these {$delim}

lint-unused-import-braces = braces around {$node} is unnecessary

lint-unused-allocation = unnecessary allocation, use `&` instead
lint-unused-allocation-mut = unnecessary allocation, use `&mut` instead

lint-builtin-while-true = denote infinite loops with `loop {"{"} ... {"}"}`
    .suggestion = use `loop`

lint-builtin-box-pointers = type uses owned (Box type) pointers: {$ty}

lint-builtin-non-shorthand-field-patterns = the `{$ident}:` in this pattern is redundant
    .suggestion = use shorthand field pattern

lint-builtin-overridden-symbol-name =
    the linker's behavior with multiple libraries exporting duplicate symbol names is undefined and Rust cannot provide guarantees when you manually override them

lint-builtin-overridden-symbol-section =
    the program's behavior with overridden link sections on items is unpredictable and Rust cannot provide guarantees when you manually override them

lint-builtin-allow-internal-unsafe =
    `allow_internal_unsafe` allows defining macros using unsafe without triggering the `unsafe_code` lint at their call site

lint-builtin-unsafe-block = usage of an `unsafe` block

lint-builtin-unsafe-trait = declaration of an `unsafe` trait

lint-builtin-unsafe-impl = implementation of an `unsafe` trait

lint-builtin-no-mangle-fn = declaration of a `no_mangle` function
lint-builtin-export-name-fn = declaration of a function with `export_name`
lint-builtin-link-section-fn = declaration of a function with `link_section`

lint-builtin-no-mangle-static = declaration of a `no_mangle` static
lint-builtin-export-name-static = declaration of a static with `export_name`
lint-builtin-link-section-static = declaration of a static with `link_section`

lint-builtin-no-mangle-method = declaration of a `no_mangle` method
lint-builtin-export-name-method = declaration of a method with `export_name`

lint-builtin-decl-unsafe-fn = declaration of an `unsafe` function
lint-builtin-decl-unsafe-method = declaration of an `unsafe` method
lint-builtin-impl-unsafe-method = implementation of an `unsafe` method

lint-builtin-missing-doc = missing documentation for {$article} {$desc}

lint-builtin-missing-copy-impl = type could implement `Copy`; consider adding `impl Copy`

lint-builtin-missing-debug-impl =
    type does not implement `{$debug}`; consider adding `#[derive(Debug)]` or a manual implementation

lint-builtin-anonymous-params = anonymous parameters are deprecated and will be removed in the next edition
    .suggestion = try naming the parameter or explicitly ignoring it

lint-builtin-deprecated-attr-link = use of deprecated attribute `{$name}`: {$reason}. See {$link}
lint-builtin-deprecated-attr-used = use of deprecated attribute `{$name}`: no longer used.
lint-builtin-deprecated-attr-default-suggestion = remove this attribute

lint-builtin-unused-doc-comment = unused doc comment
    .label = rustdoc does not generate documentation for {$kind}
    .plain-help = use `//` for a plain comment
    .block-help = use `/* */` for a plain comment

lint-builtin-no-mangle-generic = functions generic over types or consts must be mangled
    .suggestion = remove this attribute

lint-builtin-const-no-mangle = const items should never be `#[no_mangle]`
    .suggestion = try a static value

lint-builtin-mutable-transmutes =
    transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell

lint-builtin-unstable-features = unstable feature

lint-builtin-unreachable-pub = unreachable `pub` {$what}
    .suggestion = consider restricting its visibility
    .help = or consider exporting it for use by other crates

lint-builtin-type-alias-bounds-help = use fully disambiguated paths (i.e., `<T as Trait>::Assoc`) to refer to associated types in type aliases

lint-builtin-type-alias-where-clause = where clauses are not enforced in type aliases
    .suggestion = the clause will not be checked when the type alias is used, and should be removed

lint-builtin-type-alias-generic-bounds = bounds on generic parameters are not enforced in type aliases
    .suggestion = the bound will not be checked when the type alias is used, and should be removed

lint-builtin-trivial-bounds = {$predicate_kind_name} bound {$predicate} does not depend on any type or lifetime parameters

lint-builtin-ellipsis-inclusive-range-patterns = `...` range patterns are deprecated
    .suggestion = use `..=` for an inclusive range

lint-builtin-unnameable-test-items = cannot test inner items

lint-builtin-keyword-idents = `{$kw}` is a keyword in the {$next} edition
    .suggestion = you can use a raw identifier to stay compatible

lint-builtin-explicit-outlives = outlives requirements can be inferred
    .suggestion = remove {$count ->
        [one] this bound
        *[other] these bounds
    }

lint-builtin-incomplete-features = the feature `{$name}` is incomplete and may not be safe to use and/or cause compiler crashes
    .note = see issue #{$n} <https://github.com/rust-lang/rust/issues/{$n}> for more information
    .help = consider using `min_{$name}` instead, which is more stable and complete

lint-builtin-clashing-extern-same-name = `{$this_fi}` redeclared with a different signature
    .previous-decl-label = `{$orig}` previously declared here
    .mismatch-label = this signature doesn't match the previous declaration
lint-builtin-clashing-extern-diff-name = `{$this_fi}` redeclares `{$orig}` with a different signature
    .previous-decl-label = `{$orig}` previously declared here
    .mismatch-label = this signature doesn't match the previous declaration

lint-builtin-deref-nullptr = dereferencing a null pointer
    .label = this code causes undefined behavior when executed
