//! An "interner" is a data structure that associates values with usize tags and
//! allows bidirectional lookup; i.e., given a value, one can easily find the
//! type, and vice versa.

use rustc_arena::DroplessArena;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};
use rustc_macros::{symbols, HashStable_Generic};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_serialize::{UseSpecializedDecodable, UseSpecializedEncodable};

use std::cmp::{Ord, PartialEq, PartialOrd};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str;

use crate::{Span, DUMMY_SP, GLOBALS};

#[cfg(test)]
mod tests;

symbols! {
    // After modifying this list adjust `is_special`, `is_used_keyword`/`is_unused_keyword`,
    // this should be rarely necessary though if the keywords are kept in alphabetic order.
    Keywords {
        // Special reserved identifiers used internally for elided lifetimes,
        // unnamed method parameters, crate root module, error recovery etc.
        Invalid:            "",
        PathRoot:           "{{root}}",
        DollarCrate:        "$crate",
        Underscore:         "_",

        // Keywords that are used in stable Rust.
        As:                 "as",
        Break:              "break",
        Const:              "const",
        Continue:           "continue",
        Crate:              "crate",
        Else:               "else",
        Enum:               "enum",
        Extern:             "extern",
        False:              "false",
        Fn:                 "fn",
        For:                "for",
        If:                 "if",
        Impl:               "impl",
        In:                 "in",
        Let:                "let",
        Loop:               "loop",
        Match:              "match",
        Mod:                "mod",
        Move:               "move",
        Mut:                "mut",
        Pub:                "pub",
        Ref:                "ref",
        Return:             "return",
        SelfLower:          "self",
        SelfUpper:          "Self",
        Static:             "static",
        Struct:             "struct",
        Super:              "super",
        Trait:              "trait",
        True:               "true",
        Type:               "type",
        Unsafe:             "unsafe",
        Use:                "use",
        Where:              "where",
        While:              "while",

        // Keywords that are used in unstable Rust or reserved for future use.
        Abstract:           "abstract",
        Become:             "become",
        Box:                "box",
        Do:                 "do",
        Final:              "final",
        Macro:              "macro",
        Override:           "override",
        Priv:               "priv",
        Typeof:             "typeof",
        Unsized:            "unsized",
        Virtual:            "virtual",
        Yield:              "yield",

        // Edition-specific keywords that are used in stable Rust.
        Async:              "async", // >= 2018 Edition only
        Await:              "await", // >= 2018 Edition only
        Dyn:                "dyn", // >= 2018 Edition only

        // Edition-specific keywords that are used in unstable Rust or reserved for future use.
        Try:                "try", // >= 2018 Edition only

        // Special lifetime names
        UnderscoreLifetime: "'_",
        StaticLifetime:     "'static",

        // Weak keywords, have special meaning only in specific contexts.
        Auto:               "auto",
        Catch:              "catch",
        Default:            "default",
        MacroRules:         "macro_rules",
        Raw:                "raw",
        Union:              "union",
    }

    // Symbols that can be referred to with rustc_span::sym::*. The symbol is
    // the stringified identifier unless otherwise specified (e.g.
    // `proc_dash_macro` represents "proc-macro").
    //
    // As well as the symbols listed, there are symbols for the the strings
    // "0", "1", ..., "9", which are accessible via `sym::integer`.
    Symbols {
        aarch64_target_feature,
        abi,
        abi_amdgpu_kernel,
        abi_efiapi,
        abi_msp430_interrupt,
        abi_ptx,
        abi_sysv64,
        abi_thiscall,
        abi_unadjusted,
        abi_vectorcall,
        abi_x86_interrupt,
        abi_avr_interrupt,
        abort,
        aborts,
        address,
        add_with_overflow,
        advanced_slice_patterns,
        adx_target_feature,
        alias,
        align,
        alignstack,
        all,
        allocator,
        allocator_internals,
        alloc_error_handler,
        allow,
        allowed,
        allow_fail,
        allow_internal_unsafe,
        allow_internal_unstable,
        allow_internal_unstable_backcompat_hack,
        always,
        and,
        any,
        arbitrary_enum_discriminant,
        arbitrary_self_types,
        Arc,
        Arguments,
        ArgumentV1,
        arith_offset,
        arm_target_feature,
        asm,
        assert,
        associated_consts,
        associated_type_bounds,
        associated_type_defaults,
        associated_types,
        assume_init,
        async_await,
        async_closure,
        attr,
        attributes,
        attr_literals,
        att_syntax,
        augmented_assignments,
        automatically_derived,
        avx512_target_feature,
        await_macro,
        begin_panic,
        bench,
        bin,
        bind_by_move_pattern_guards,
        bindings_after_at,
        block,
        bool,
        borrowck_graphviz_format,
        borrowck_graphviz_postflow,
        borrowck_graphviz_preflow,
        box_patterns,
        box_syntax,
        braced_empty_structs,
        bswap,
        bitreverse,
        C,
        caller_location,
        cdylib,
        cfg,
        cfg_accessible,
        cfg_attr,
        cfg_attr_multi,
        cfg_doctest,
        cfg_sanitize,
        cfg_target_feature,
        cfg_target_has_atomic,
        cfg_target_thread_local,
        cfg_target_vendor,
        cfg_version,
        char,
        clippy,
        clone,
        Clone,
        clone_closures,
        clone_from,
        closure_to_fn_coercion,
        cmp,
        cmpxchg16b_target_feature,
        cold,
        column,
        compile_error,
        compiler_builtins,
        concat,
        concat_idents,
        conservative_impl_trait,
        console,
        const_compare_raw_pointers,
        const_constructor,
        const_eval_limit,
        const_extern_fn,
        const_fn,
        const_fn_union,
        const_generics,
        const_if_match,
        const_indexing,
        const_in_array_repeat_expressions,
        const_let,
        const_loop,
        const_mut_refs,
        const_panic,
        const_precise_live_drops,
        const_raw_ptr_deref,
        const_raw_ptr_to_usize_cast,
        const_transmute,
        const_trait_bound_opt_out,
        const_trait_impl,
        contents,
        context,
        convert,
        Copy,
        copy_closures,
        core,
        core_intrinsics,
        crate_id,
        crate_in_paths,
        crate_local,
        crate_name,
        crate_type,
        crate_visibility_modifier,
        ctpop,
        cttz,
        cttz_nonzero,
        ctlz,
        ctlz_nonzero,
        custom_attribute,
        custom_derive,
        custom_inner_attributes,
        custom_test_frameworks,
        c_variadic,
        debug_trait,
        declare_lint_pass,
        decl_macro,
        debug,
        Debug,
        Decodable,
        Default,
        default_lib_allocator,
        default_type_parameter_fallback,
        default_type_params,
        delay_span_bug_from_inside_query,
        deny,
        deprecated,
        deref,
        deref_mut,
        derive,
        diagnostic,
        direct,
        discriminant_value,
        doc,
        doc_alias,
        doc_cfg,
        doc_keyword,
        doc_masked,
        doctest,
        document_private_items,
        dotdoteq_in_patterns,
        dotdot_in_tuple_patterns,
        double_braced_crate: "{{crate}}",
        double_braced_impl: "{{impl}}",
        double_braced_misc: "{{misc}}",
        double_braced_closure: "{{closure}}",
        double_braced_constructor: "{{constructor}}",
        double_braced_constant: "{{constant}}",
        double_braced_opaque: "{{opaque}}",
        dropck_eyepatch,
        dropck_parametricity,
        drop_types_in_const,
        dylib,
        dyn_trait,
        eh_personality,
        enable,
        Encodable,
        env,
        eq,
        err,
        Err,
        Eq,
        Equal,
        enclosing_scope,
        except,
        exclusive_range_pattern,
        exhaustive_integer_patterns,
        exhaustive_patterns,
        existential_type,
        expected,
        export_name,
        expr,
        extern_absolute_paths,
        external_doc,
        extern_crate_item_prelude,
        extern_crate_self,
        extern_in_paths,
        extern_prelude,
        extern_types,
        f16c_target_feature,
        f32,
        f64,
        feature,
        ffi_const,
        ffi_pure,
        ffi_returns_twice,
        field,
        field_init_shorthand,
        file,
        fmt,
        fmt_internals,
        fn_must_use,
        forbid,
        format_args,
        format_args_nl,
        from,
        From,
        from_desugaring,
        from_error,
        from_generator,
        from_method,
        from_ok,
        from_usize,
        fundamental,
        future,
        Future,
        FxHashSet,
        FxHashMap,
        gen_future,
        gen_kill,
        generators,
        generic_associated_types,
        generic_param_attrs,
        get_context,
        global_allocator,
        global_asm,
        globs,
        half_open_range_patterns,
        hash,
        Hash,
        HashSet,
        HashMap,
        hexagon_target_feature,
        hidden,
        homogeneous_aggregate,
        html_favicon_url,
        html_logo_url,
        html_no_source,
        html_playground_url,
        html_root_url,
        i128,
        i128_type,
        i16,
        i32,
        i64,
        i8,
        ident,
        if_let,
        if_while_or_patterns,
        ignore,
        inlateout,
        inout,
        impl_header_lifetime_elision,
        impl_lint_pass,
        impl_trait_in_bindings,
        import_shadowing,
        index,
        index_mut,
        in_band_lifetimes,
        include,
        include_bytes,
        include_str,
        inclusive_range_syntax,
        infer_outlives_requirements,
        infer_static_outlives_requirements,
        inline,
        intel,
        into_iter,
        IntoIterator,
        into_result,
        intrinsics,
        irrefutable_let_patterns,
        isize,
        issue,
        issue_5723_bootstrap,
        issue_tracker_base_url,
        item,
        item_context: "ItemContext",
        item_like_imports,
        iter,
        Iterator,
        keyword,
        kind,
        label,
        label_break_value,
        lang,
        lang_items,
        lateout,
        let_chains,
        lhs,
        lib,
        lifetime,
        line,
        link,
        linkage,
        link_args,
        link_cfg,
        link_llvm_intrinsics,
        link_name,
        link_ordinal,
        link_section,
        LintPass,
        lint_reasons,
        literal,
        llvm_asm,
        local_inner_macros,
        log_syntax,
        loop_break_value,
        macro_at_most_once_rep,
        macro_escape,
        macro_export,
        macro_lifetime_matcher,
        macro_literal_matcher,
        macro_reexport,
        macros_in_extern,
        macro_use,
        macro_vis_matcher,
        main,
        managed_boxes,
        marker,
        marker_trait_attr,
        masked,
        match_beginning_vert,
        match_default_bindings,
        may_dangle,
        maybe_uninit_uninit,
        maybe_uninit_zeroed,
        mem_uninitialized,
        mem_zeroed,
        member_constraints,
        memory,
        message,
        meta,
        min_align_of,
        min_const_fn,
        min_const_unsafe_fn,
        min_specialization,
        mips_target_feature,
        mmx_target_feature,
        module,
        module_path,
        more_struct_aliases,
        move_ref_pattern,
        move_val_init,
        movbe_target_feature,
        mul_with_overflow,
        must_use,
        naked,
        naked_functions,
        name,
        needs_allocator,
        needs_drop,
        needs_panic_runtime,
        negate_unsigned,
        negative_impls,
        never,
        never_type,
        never_type_fallback,
        new,
        next,
        __next,
        nll,
        no_builtins,
        no_core,
        no_crate_inject,
        no_debug,
        no_default_passes,
        no_implicit_prelude,
        no_inline,
        no_link,
        no_main,
        no_mangle,
        nomem,
        non_ascii_idents,
        None,
        non_exhaustive,
        non_modrs_mods,
        noreturn,
        no_niche,
        no_sanitize,
        nostack,
        no_stack_check,
        no_start,
        no_std,
        not,
        note,
        object_safe_for_dispatch,
        offset,
        Ok,
        omit_gdb_pretty_printer_section,
        on,
        on_unimplemented,
        oom,
        ops,
        optimize,
        optimize_attribute,
        optin_builtin_traits,
        option,
        Option,
        option_env,
        options,
        opt_out_copy,
        or,
        or_patterns,
        Ord,
        Ordering,
        out,
        Output,
        overlapping_marker_traits,
        packed,
        panic,
        panic_handler,
        panic_impl,
        panic_implementation,
        panic_runtime,
        parent_trait,
        partial_cmp,
        param_attrs,
        PartialEq,
        PartialOrd,
        passes,
        pat,
        path,
        pattern_parentheses,
        Pending,
        pin,
        Pin,
        pinned,
        platform_intrinsics,
        plugin,
        plugin_registrar,
        plugins,
        poll,
        Poll,
        powerpc_target_feature,
        precise_pointer_size_matching,
        pref_align_of,
        prelude,
        prelude_import,
        preserves_flags,
        primitive,
        proc_dash_macro: "proc-macro",
        proc_macro,
        proc_macro_attribute,
        proc_macro_def_site,
        proc_macro_derive,
        proc_macro_expr,
        proc_macro_gen,
        proc_macro_hygiene,
        proc_macro_internals,
        proc_macro_mod,
        proc_macro_non_items,
        proc_macro_path_invoc,
        profiler_runtime,
        ptr_offset_from,
        pub_restricted,
        pure,
        pushpop_unsafe,
        quad_precision_float,
        question_mark,
        quote,
        Range,
        RangeFrom,
        RangeFull,
        RangeInclusive,
        RangeTo,
        RangeToInclusive,
        raw_dylib,
        raw_identifiers,
        raw_ref_op,
        Rc,
        readonly,
        Ready,
        reason,
        recursion_limit,
        reexport_test_harness_main,
        reflect,
        register_attr,
        register_tool,
        relaxed_adts,
        repr,
        repr128,
        repr_align,
        repr_align_enum,
        repr_no_niche,
        repr_packed,
        repr_simd,
        repr_transparent,
        re_rebalance_coherence,
        result,
        Result,
        Return,
        rhs,
        riscv_target_feature,
        rlib,
        rotate_left,
        rotate_right,
        rt,
        rtm_target_feature,
        rust,
        rust_2015_preview,
        rust_2018_preview,
        rust_begin_unwind,
        rustc,
        RustcDecodable,
        RustcEncodable,
        rustc_allocator,
        rustc_allocator_nounwind,
        rustc_allow_const_fn_ptr,
        rustc_args_required_const,
        rustc_attrs,
        rustc_builtin_macro,
        rustc_clean,
        rustc_const_unstable,
        rustc_const_stable,
        rustc_conversion_suggestion,
        rustc_def_path,
        rustc_deprecated,
        rustc_diagnostic_item,
        rustc_diagnostic_macros,
        rustc_dirty,
        rustc_dummy,
        rustc_dump_env_program_clauses,
        rustc_dump_program_clauses,
        rustc_dump_user_substs,
        rustc_error,
        rustc_expected_cgu_reuse,
        rustc_if_this_changed,
        rustc_inherit_overflow_checks,
        rustc_layout,
        rustc_layout_scalar_valid_range_end,
        rustc_layout_scalar_valid_range_start,
        rustc_macro_transparency,
        rustc_mir,
        rustc_nonnull_optimization_guaranteed,
        rustc_object_lifetime_default,
        rustc_on_unimplemented,
        rustc_outlives,
        rustc_paren_sugar,
        rustc_partition_codegened,
        rustc_partition_reused,
        rustc_peek,
        rustc_peek_definite_init,
        rustc_peek_liveness,
        rustc_peek_maybe_init,
        rustc_peek_maybe_uninit,
        rustc_peek_indirectly_mutable,
        rustc_private,
        rustc_proc_macro_decls,
        rustc_promotable,
        rustc_regions,
        rustc_unsafe_specialization_marker,
        rustc_specialization_trait,
        rustc_stable,
        rustc_std_internal_symbol,
        rustc_symbol_name,
        rustc_synthetic,
        rustc_reservation_impl,
        rustc_test_marker,
        rustc_then_this_would_need,
        rustc_variance,
        rustfmt,
        rust_eh_personality,
        rust_oom,
        rvalue_static_promotion,
        sanitize,
        sanitizer_runtime,
        saturating_add,
        saturating_sub,
        _Self,
        self_in_typedefs,
        self_struct_ctor,
        send_trait,
        should_panic,
        simd,
        simd_extract,
        simd_ffi,
        simd_insert,
        since,
        size,
        size_of,
        slice_patterns,
        slicing_syntax,
        soft,
        Some,
        specialization,
        speed,
        sse4a_target_feature,
        stable,
        staged_api,
        start,
        static_in_const,
        staticlib,
        static_nobundle,
        static_recursion,
        std,
        std_inject,
        str,
        stringify,
        stmt,
        stmt_expr_attributes,
        stop_after_dataflow,
        struct_field_attributes,
        struct_inherit,
        structural_match,
        struct_variant,
        sty,
        sub_with_overflow,
        suggestion,
        sym,
        sync_trait,
        target_feature,
        target_feature_11,
        target_has_atomic,
        target_has_atomic_load_store,
        target_thread_local,
        task,
        _task_context,
        tbm_target_feature,
        termination_trait,
        termination_trait_test,
        test,
        test_2018_feature,
        test_accepted_feature,
        test_case,
        test_removed_feature,
        test_runner,
        then_with,
        thread,
        thread_local,
        tool_attributes,
        tool_lints,
        trace_macros,
        track_caller,
        trait_alias,
        transmute,
        transparent,
        transparent_enums,
        transparent_unions,
        trivial_bounds,
        Try,
        try_blocks,
        try_trait,
        tt,
        tuple_indexing,
        two_phase,
        Ty,
        ty,
        type_alias_impl_trait,
        type_id,
        type_name,
        TyCtxt,
        TyKind,
        type_alias_enum_variants,
        type_ascription,
        type_length_limit,
        type_macros,
        u128,
        u16,
        u32,
        u64,
        u8,
        unboxed_closures,
        unchecked_add,
        unchecked_div,
        unchecked_mul,
        unchecked_rem,
        unchecked_shl,
        unchecked_shr,
        unchecked_sub,
        underscore_const_names,
        underscore_imports,
        underscore_lifetimes,
        uniform_paths,
        universal_impl_trait,
        unmarked_api,
        unreachable_code,
        unrestricted_attribute_tokens,
        unsafe_block_in_unsafe_fn,
        unsafe_no_drop_flag,
        unsized_locals,
        unsized_tuple_coercion,
        unstable,
        untagged_unions,
        unwind,
        unwind_attributes,
        unwrap_or,
        used,
        use_extern_macros,
        use_nested_groups,
        usize,
        v1,
        val,
        var,
        vec,
        Vec,
        version,
        vis,
        visible_private_types,
        volatile,
        warn,
        wasm_import_module,
        wasm_target_feature,
        while_let,
        windows,
        windows_subsystem,
        wrapping_add,
        wrapping_sub,
        wrapping_mul,
        Yield,
    }
}

#[derive(Copy, Clone, Eq, HashStable_Generic)]
pub struct Ident {
    pub name: Symbol,
    pub span: Span,
}

impl Ident {
    #[inline]
    /// Constructs a new identifier from a symbol and a span.
    pub const fn new(name: Symbol, span: Span) -> Ident {
        Ident { name, span }
    }

    /// Constructs a new identifier with a dummy span.
    #[inline]
    pub const fn with_dummy_span(name: Symbol) -> Ident {
        Ident::new(name, DUMMY_SP)
    }

    #[inline]
    pub fn invalid() -> Ident {
        Ident::with_dummy_span(kw::Invalid)
    }

    /// Maps a string to an identifier with a dummy span.
    pub fn from_str(string: &str) -> Ident {
        Ident::with_dummy_span(Symbol::intern(string))
    }

    /// Maps a string and a span to an identifier.
    pub fn from_str_and_span(string: &str, span: Span) -> Ident {
        Ident::new(Symbol::intern(string), span)
    }

    /// Replaces `lo` and `hi` with those from `span`, but keep hygiene context.
    pub fn with_span_pos(self, span: Span) -> Ident {
        Ident::new(self.name, span.with_ctxt(self.span.ctxt()))
    }

    pub fn without_first_quote(self) -> Ident {
        Ident::new(Symbol::intern(self.as_str().trim_start_matches('\'')), self.span)
    }

    /// "Normalize" ident for use in comparisons using "item hygiene".
    /// Identifiers with same string value become same if they came from the same macro 2.0 macro
    /// (e.g., `macro` item, but not `macro_rules` item) and stay different if they came from
    /// different macro 2.0 macros.
    /// Technically, this operation strips all non-opaque marks from ident's syntactic context.
    pub fn normalize_to_macros_2_0(self) -> Ident {
        Ident::new(self.name, self.span.normalize_to_macros_2_0())
    }

    /// "Normalize" ident for use in comparisons using "local variable hygiene".
    /// Identifiers with same string value become same if they came from the same non-transparent
    /// macro (e.g., `macro` or `macro_rules!` items) and stay different if they came from different
    /// non-transparent macros.
    /// Technically, this operation strips all transparent marks from ident's syntactic context.
    pub fn normalize_to_macro_rules(self) -> Ident {
        Ident::new(self.name, self.span.normalize_to_macro_rules())
    }

    /// Convert the name to a `SymbolStr`. This is a slowish operation because
    /// it requires locking the symbol interner.
    pub fn as_str(self) -> SymbolStr {
        self.name.as_str()
    }
}

impl PartialEq for Ident {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name && self.span.ctxt() == rhs.span.ctxt()
    }
}

impl Hash for Ident {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.span.ctxt().hash(state);
    }
}

impl fmt::Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)?;
        fmt::Debug::fmt(&self.span.ctxt(), f)
    }
}

/// This implementation is supposed to be used in error messages, so it's expected to be identical
/// to printing the original identifier token written in source code (`token_to_string`),
/// except that AST identifiers don't keep the rawness flag, so we have to guess it.
impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&IdentPrinter::new(self.name, self.is_raw_guess(), None), f)
    }
}

impl UseSpecializedEncodable for Ident {
    fn default_encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_struct("Ident", 2, |s| {
            s.emit_struct_field("name", 0, |s| self.name.encode(s))?;
            s.emit_struct_field("span", 1, |s| self.span.encode(s))
        })
    }
}

impl UseSpecializedDecodable for Ident {
    fn default_decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
        d.read_struct("Ident", 2, |d| {
            Ok(Ident {
                name: d.read_struct_field("name", 0, Decodable::decode)?,
                span: d.read_struct_field("span", 1, Decodable::decode)?,
            })
        })
    }
}

/// This is the most general way to print identifiers.
/// AST pretty-printer is used as a fallback for turning AST structures into token streams for
/// proc macros. Additionally, proc macros may stringify their input and expect it survive the
/// stringification (especially true for proc macro derives written between Rust 1.15 and 1.30).
/// So we need to somehow pretty-print `$crate` in a way preserving at least some of its
/// hygiene data, most importantly name of the crate it refers to.
/// As a result we print `$crate` as `crate` if it refers to the local crate
/// and as `::other_crate_name` if it refers to some other crate.
/// Note, that this is only done if the ident token is printed from inside of AST pretty-pringing,
/// but not otherwise. Pretty-printing is the only way for proc macros to discover token contents,
/// so we should not perform this lossy conversion if the top level call to the pretty-printer was
/// done for a token stream or a single token.
pub struct IdentPrinter {
    symbol: Symbol,
    is_raw: bool,
    /// Span used for retrieving the crate name to which `$crate` refers to,
    /// if this field is `None` then the `$crate` conversion doesn't happen.
    convert_dollar_crate: Option<Span>,
}

impl IdentPrinter {
    /// The most general `IdentPrinter` constructor. Do not use this.
    pub fn new(symbol: Symbol, is_raw: bool, convert_dollar_crate: Option<Span>) -> IdentPrinter {
        IdentPrinter { symbol, is_raw, convert_dollar_crate }
    }

    /// This implementation is supposed to be used when printing identifiers
    /// as a part of pretty-printing for larger AST pieces.
    /// Do not use this either.
    pub fn for_ast_ident(ident: Ident, is_raw: bool) -> IdentPrinter {
        IdentPrinter::new(ident.name, is_raw, Some(ident.span))
    }
}

impl fmt::Display for IdentPrinter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_raw {
            f.write_str("r#")?;
        } else {
            if self.symbol == kw::DollarCrate {
                if let Some(span) = self.convert_dollar_crate {
                    let converted = span.ctxt().dollar_crate_name();
                    if !converted.is_path_segment_keyword() {
                        f.write_str("::")?;
                    }
                    return fmt::Display::fmt(&converted, f);
                }
            }
        }
        fmt::Display::fmt(&self.symbol, f)
    }
}

/// An newtype around `Ident` that calls [Ident::normalize_to_macro_rules] on
/// construction.
// FIXME(matthewj, petrochenkov) Use this more often, add a similar
// `ModernIdent` struct and use that as well.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct MacroRulesNormalizedIdent(Ident);

impl MacroRulesNormalizedIdent {
    pub fn new(ident: Ident) -> Self {
        Self(ident.normalize_to_macro_rules())
    }
}

impl fmt::Debug for MacroRulesNormalizedIdent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for MacroRulesNormalizedIdent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// An interned string.
///
/// Internally, a `Symbol` is implemented as an index, and all operations
/// (including hashing, equality, and ordering) operate on that index. The use
/// of `rustc_index::newtype_index!` means that `Option<Symbol>` only takes up 4 bytes,
/// because `rustc_index::newtype_index!` reserves the last 256 values for tagging purposes.
///
/// Note that `Symbol` cannot directly be a `rustc_index::newtype_index!` because it
/// implements `fmt::Debug`, `Encodable`, and `Decodable` in special ways.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(SymbolIndex);

rustc_index::newtype_index! {
    pub struct SymbolIndex { .. }
}

impl Symbol {
    const fn new(n: u32) -> Self {
        Symbol(SymbolIndex::from_u32(n))
    }

    /// Maps a string to its interned representation.
    pub fn intern(string: &str) -> Self {
        with_interner(|interner| interner.intern(string))
    }

    /// Access the symbol's chars. This is a slowish operation because it
    /// requires locking the symbol interner.
    pub fn with<F: FnOnce(&str) -> R, R>(self, f: F) -> R {
        with_interner(|interner| f(interner.get(self)))
    }

    /// Convert to a `SymbolStr`. This is a slowish operation because it
    /// requires locking the symbol interner.
    pub fn as_str(self) -> SymbolStr {
        with_interner(|interner| unsafe {
            SymbolStr { string: std::mem::transmute::<&str, &str>(interner.get(self)) }
        })
    }

    pub fn as_u32(self) -> u32 {
        self.0.as_u32()
    }

    /// This method is supposed to be used in error messages, so it's expected to be
    /// identical to printing the original identifier token written in source code
    /// (`token_to_string`, `Ident::to_string`), except that symbols don't keep the rawness flag
    /// or edition, so we have to guess the rawness using the global edition.
    pub fn to_ident_string(self) -> String {
        Ident::with_dummy_span(self).to_string()
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.with(|str| fmt::Debug::fmt(&str, f))
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.with(|str| fmt::Display::fmt(&str, f))
    }
}

impl Encodable for Symbol {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        self.with(|string| s.emit_str(string))
    }
}

impl Decodable for Symbol {
    #[inline]
    fn decode<D: Decoder>(d: &mut D) -> Result<Symbol, D::Error> {
        Ok(Symbol::intern(&d.read_str()?))
    }
}

impl<CTX> HashStable<CTX> for Symbol {
    #[inline]
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.as_str().hash_stable(hcx, hasher);
    }
}

impl<CTX> ToStableHashKey<CTX> for Symbol {
    type KeyType = SymbolStr;

    #[inline]
    fn to_stable_hash_key(&self, _: &CTX) -> SymbolStr {
        self.as_str()
    }
}

// The `&'static str`s in this type actually point into the arena.
#[derive(Default)]
pub struct Interner {
    arena: DroplessArena,
    names: FxHashMap<&'static str, Symbol>,
    strings: Vec<&'static str>,
}

impl Interner {
    fn prefill(init: &[&'static str]) -> Self {
        Interner {
            strings: init.into(),
            names: init.iter().copied().zip((0..).map(Symbol::new)).collect(),
            ..Default::default()
        }
    }

    #[inline]
    pub fn intern(&mut self, string: &str) -> Symbol {
        if let Some(&name) = self.names.get(string) {
            return name;
        }

        let name = Symbol::new(self.strings.len() as u32);

        // `from_utf8_unchecked` is safe since we just allocated a `&str` which is known to be
        // UTF-8.
        let string: &str =
            unsafe { str::from_utf8_unchecked(self.arena.alloc_slice(string.as_bytes())) };
        // It is safe to extend the arena allocation to `'static` because we only access
        // these while the arena is still alive.
        let string: &'static str = unsafe { &*(string as *const str) };
        self.strings.push(string);
        self.names.insert(string, name);
        name
    }

    // Get the symbol as a string. `Symbol::as_str()` should be used in
    // preference to this function.
    pub fn get(&self, symbol: Symbol) -> &str {
        self.strings[symbol.0.as_usize()]
    }
}

// This module has a very short name because it's used a lot.
/// This module contains all the defined keyword `Symbol`s.
///
/// Given that `kw` is imported, use them like `kw::keyword_name`.
/// For example `kw::Loop` or `kw::Break`.
pub mod kw {
    use super::Symbol;
    keywords!();
}

// This module has a very short name because it's used a lot.
/// This module contains all the defined non-keyword `Symbol`s.
///
/// Given that `sym` is imported, use them like `sym::symbol_name`.
/// For example `sym::rustfmt` or `sym::u8`.
#[allow(rustc::default_hash_types)]
pub mod sym {
    use super::Symbol;
    use std::convert::TryInto;

    symbols!();

    // Used from a macro in `librustc_feature/accepted.rs`
    pub use super::kw::MacroRules as macro_rules;

    // Get the symbol for an integer. The first few non-negative integers each
    // have a static symbol and therefore are fast.
    pub fn integer<N: TryInto<usize> + Copy + ToString>(n: N) -> Symbol {
        if let Result::Ok(idx) = n.try_into() {
            if let Option::Some(&sym_) = digits_array.get(idx) {
                return sym_;
            }
        }
        Symbol::intern(&n.to_string())
    }
}

impl Symbol {
    fn is_used_keyword_2018(self) -> bool {
        self >= kw::Async && self <= kw::Dyn
    }

    fn is_unused_keyword_2018(self) -> bool {
        self == kw::Try
    }

    /// Used for sanity checking rustdoc keyword sections.
    pub fn is_doc_keyword(self) -> bool {
        self <= kw::Union
    }

    /// A keyword or reserved identifier that can be used as a path segment.
    pub fn is_path_segment_keyword(self) -> bool {
        self == kw::Super
            || self == kw::SelfLower
            || self == kw::SelfUpper
            || self == kw::Crate
            || self == kw::PathRoot
            || self == kw::DollarCrate
    }

    /// Returns `true` if the symbol is `true` or `false`.
    pub fn is_bool_lit(self) -> bool {
        self == kw::True || self == kw::False
    }

    /// This symbol can be a raw identifier.
    pub fn can_be_raw(self) -> bool {
        self != kw::Invalid && self != kw::Underscore && !self.is_path_segment_keyword()
    }
}

impl Ident {
    // Returns `true` for reserved identifiers used internally for elided lifetimes,
    // unnamed method parameters, crate root module, error recovery etc.
    pub fn is_special(self) -> bool {
        self.name <= kw::Underscore
    }

    /// Returns `true` if the token is a keyword used in the language.
    pub fn is_used_keyword(self) -> bool {
        // Note: `span.edition()` is relatively expensive, don't call it unless necessary.
        self.name >= kw::As && self.name <= kw::While
            || self.name.is_used_keyword_2018() && self.span.rust_2018()
    }

    /// Returns `true` if the token is a keyword reserved for possible future use.
    pub fn is_unused_keyword(self) -> bool {
        // Note: `span.edition()` is relatively expensive, don't call it unless necessary.
        self.name >= kw::Abstract && self.name <= kw::Yield
            || self.name.is_unused_keyword_2018() && self.span.rust_2018()
    }

    /// Returns `true` if the token is either a special identifier or a keyword.
    pub fn is_reserved(self) -> bool {
        self.is_special() || self.is_used_keyword() || self.is_unused_keyword()
    }

    /// A keyword or reserved identifier that can be used as a path segment.
    pub fn is_path_segment_keyword(self) -> bool {
        self.name.is_path_segment_keyword()
    }

    /// We see this identifier in a normal identifier position, like variable name or a type.
    /// How was it written originally? Did it use the raw form? Let's try to guess.
    pub fn is_raw_guess(self) -> bool {
        self.name.can_be_raw() && self.is_reserved()
    }
}

#[inline]
fn with_interner<T, F: FnOnce(&mut Interner) -> T>(f: F) -> T {
    GLOBALS.with(|globals| f(&mut *globals.symbol_interner.lock()))
}

/// An alternative to `Symbol`, useful when the chars within the symbol need to
/// be accessed. It deliberately has limited functionality and should only be
/// used for temporary values.
///
/// Because the interner outlives any thread which uses this type, we can
/// safely treat `string` which points to interner data, as an immortal string,
/// as long as this type never crosses between threads.
//
// FIXME: ensure that the interner outlives any thread which uses `SymbolStr`,
// by creating a new thread right after constructing the interner.
#[derive(Clone, Eq, PartialOrd, Ord)]
pub struct SymbolStr {
    string: &'static str,
}

// This impl allows a `SymbolStr` to be directly equated with a `String` or
// `&str`.
impl<T: std::ops::Deref<Target = str>> std::cmp::PartialEq<T> for SymbolStr {
    fn eq(&self, other: &T) -> bool {
        self.string == other.deref()
    }
}

impl !Send for SymbolStr {}
impl !Sync for SymbolStr {}

/// This impl means that if `ss` is a `SymbolStr`:
/// - `*ss` is a `str`;
/// - `&*ss` is a `&str`;
/// - `&ss as &str` is a `&str`, which means that `&ss` can be passed to a
///   function expecting a `&str`.
impl std::ops::Deref for SymbolStr {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.string
    }
}

impl fmt::Debug for SymbolStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.string, f)
    }
}

impl fmt::Display for SymbolStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.string, f)
    }
}

impl<CTX> HashStable<CTX> for SymbolStr {
    #[inline]
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.string.hash_stable(hcx, hasher)
    }
}

impl<CTX> ToStableHashKey<CTX> for SymbolStr {
    type KeyType = SymbolStr;

    #[inline]
    fn to_stable_hash_key(&self, _: &CTX) -> SymbolStr {
        self.clone()
    }
}
