//! An "interner" is a data structure that associates values with usize tags and
//! allows bidirectional lookup; i.e., given a value, one can easily find the
//! type, and vice versa.

use arena::DroplessArena;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::newtype_index;
use rustc_macros::symbols;
use serialize::{Decodable, Decoder, Encodable, Encoder};

use std::cmp::{PartialEq, Ordering, PartialOrd, Ord};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str;

use crate::hygiene::SyntaxContext;
use crate::{Span, DUMMY_SP, GLOBALS};

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
        Dyn:                "dyn", // >= 2018 Edition only

        // Edition-specific keywords that are used in unstable Rust or reserved for future use.
        Async:              "async", // >= 2018 Edition only
        Await:              "await", // >= 2018 Edition only
        Try:                "try", // >= 2018 Edition only

        // Special lifetime names
        UnderscoreLifetime: "'_",
        StaticLifetime:     "'static",

        // Weak keywords, have special meaning only in specific contexts.
        Auto:               "auto",
        Catch:              "catch",
        Default:            "default",
        Existential:        "existential",
        Union:              "union",
    }

    // Symbols that can be referred to with syntax_pos::sym::*. The symbol is
    // the stringified identifier unless otherwise specified (e.g.
    // `proc_dash_macro` represents "proc-macro").
    //
    // As well as the symbols listed, there are symbols for the the strings
    // "0", "1", ..., "9", which are accessible via `sym::integer`.
    Symbols {
        aarch64_target_feature,
        abi,
        abi_amdgpu_kernel,
        abi_msp430_interrupt,
        abi_ptx,
        abi_sysv64,
        abi_thiscall,
        abi_unadjusted,
        abi_vectorcall,
        abi_x86_interrupt,
        aborts,
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
        Arguments,
        ArgumentV1,
        arm_target_feature,
        asm,
        associated_consts,
        associated_type_bounds,
        associated_type_defaults,
        associated_types,
        async_await,
        async_closure,
        attr,
        attributes,
        attr_literals,
        augmented_assignments,
        automatically_derived,
        avx512_target_feature,
        await_macro,
        begin_panic,
        bench,
        bin,
        bind_by_move_pattern_guards,
        block,
        bool,
        borrowck_graphviz_postflow,
        borrowck_graphviz_preflow,
        box_patterns,
        box_syntax,
        braced_empty_structs,
        C,
        cdylib,
        cfg,
        cfg_attr,
        cfg_attr_multi,
        cfg_target_feature,
        cfg_target_has_atomic,
        cfg_target_thread_local,
        cfg_target_vendor,
        char,
        clone,
        Clone,
        clone_closures,
        clone_from,
        closure_to_fn_coercion,
        cmp,
        cmpxchg16b_target_feature,
        cold,
        compile_error,
        compiler_builtins,
        concat_idents,
        conservative_impl_trait,
        console,
        const_compare_raw_pointers,
        const_constructor,
        const_fn,
        const_fn_union,
        const_generics,
        const_indexing,
        const_let,
        const_panic,
        const_raw_ptr_deref,
        const_raw_ptr_to_usize_cast,
        const_transmute,
        contents,
        context,
        convert,
        copy_closures,
        core,
        core_intrinsics,
        crate_id,
        crate_in_paths,
        crate_local,
        crate_name,
        crate_type,
        crate_visibility_modifier,
        custom_attribute,
        custom_derive,
        custom_inner_attributes,
        custom_test_frameworks,
        c_variadic,
        decl_macro,
        Default,
        default_lib_allocator,
        default_type_parameter_fallback,
        default_type_params,
        deny,
        deprecated,
        deref,
        deref_mut,
        derive,
        direct,
        doc,
        doc_alias,
        doc_cfg,
        doc_keyword,
        doc_masked,
        doc_spotlight,
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
        eh_unwind_resume,
        enable,
        eq,
        err,
        Err,
        Equal,
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
        ffi_returns_twice,
        field,
        field_init_shorthand,
        file,
        fmt,
        fmt_internals,
        fn_must_use,
        forbid,
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
        generators,
        generic_associated_types,
        generic_param_attrs,
        global_allocator,
        global_asm,
        globs,
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
        impl_header_lifetime_elision,
        impl_trait_in_bindings,
        import_shadowing,
        index,
        index_mut,
        in_band_lifetimes,
        include,
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
        item_like_imports,
        iter,
        Iterator,
        keyword,
        kind,
        label,
        label_break_value,
        lang,
        lang_items,
        let_chains,
        lhs,
        lib,
        lifetime,
        link,
        linkage,
        link_args,
        link_cfg,
        link_llvm_intrinsics,
        link_name,
        link_section,
        lint_reasons,
        literal,
        local_inner_macros,
        log_syntax,
        loop_break_value,
        macro_at_most_once_rep,
        macro_escape,
        macro_export,
        macro_lifetime_matcher,
        macro_literal_matcher,
        macro_reexport,
        macro_rules,
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
        member_constraints,
        message,
        meta,
        min_const_fn,
        min_const_unsafe_fn,
        mips_target_feature,
        mmx_target_feature,
        module,
        more_struct_aliases,
        movbe_target_feature,
        must_use,
        naked,
        naked_functions,
        name,
        needs_allocator,
        needs_panic_runtime,
        negate_unsigned,
        never,
        never_type,
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
        non_ascii_idents,
        None,
        non_exhaustive,
        non_modrs_mods,
        no_stack_check,
        no_start,
        no_std,
        not,
        note,
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
        opt_out_copy,
        or,
        Ord,
        Ordering,
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
        Poll,
        poll_with_tls_context,
        powerpc_target_feature,
        precise_pointer_size_matching,
        prelude,
        prelude_import,
        primitive,
        proc_dash_macro: "proc-macro",
        proc_macro,
        proc_macro_attribute,
        proc_macro_def_site,
        proc_macro_derive,
        proc_macro_expr,
        proc_macro_gen,
        proc_macro_hygiene,
        proc_macro_mod,
        proc_macro_non_items,
        proc_macro_path_invoc,
        profiler_runtime,
        pub_restricted,
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
        raw_identifiers,
        Ready,
        reason,
        recursion_limit,
        reexport_test_harness_main,
        reflect,
        relaxed_adts,
        repr,
        repr128,
        repr_align,
        repr_align_enum,
        repr_packed,
        repr_simd,
        repr_transparent,
        re_rebalance_coherence,
        result,
        Result,
        Return,
        rhs,
        rlib,
        rt,
        rtm_target_feature,
        rust,
        rust_2015_preview,
        rust_2018_preview,
        rust_begin_unwind,
        rustc,
        rustc_allocator,
        rustc_allocator_nounwind,
        rustc_allow_const_fn_ptr,
        rustc_args_required_const,
        rustc_attrs,
        rustc_clean,
        rustc_const_unstable,
        rustc_conversion_suggestion,
        rustc_copy_clone_marker,
        rustc_def_path,
        rustc_deprecated,
        rustc_diagnostic_macros,
        rustc_dirty,
        rustc_doc_only_macro,
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
        rustc_peek_maybe_init,
        rustc_peek_maybe_uninit,
        rustc_private,
        rustc_proc_macro_decls,
        rustc_promotable,
        rustc_regions,
        rustc_stable,
        rustc_std_internal_symbol,
        rustc_symbol_name,
        rustc_synthetic,
        rustc_test_marker,
        rustc_then_this_would_need,
        rustc_transparent_macro,
        rustc_variance,
        rustdoc,
        rust_eh_personality,
        rust_eh_unwind_resume,
        rust_oom,
        __rust_unstable_column,
        rvalue_static_promotion,
        sanitizer_runtime,
        _Self,
        self_in_typedefs,
        self_struct_ctor,
        Send,
        should_panic,
        simd,
        simd_ffi,
        since,
        size,
        slice_patterns,
        slicing_syntax,
        Some,
        specialization,
        speed,
        spotlight,
        sse4a_target_feature,
        stable,
        staged_api,
        start,
        static_in_const,
        staticlib,
        static_nobundle,
        static_recursion,
        std,
        str,
        stmt,
        stmt_expr_attributes,
        stop_after_dataflow,
        struct_field_attributes,
        struct_inherit,
        structural_match,
        struct_variant,
        sty,
        suggestion,
        target_feature,
        target_has_atomic,
        target_thread_local,
        task,
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
        thread_local,
        tool_attributes,
        tool_lints,
        trace_macros,
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
        Ty,
        ty,
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
        underscore_const_names,
        underscore_imports,
        underscore_lifetimes,
        uniform_paths,
        universal_impl_trait,
        unmarked_api,
        unreachable_code,
        unrestricted_attribute_tokens,
        unsafe_destructor_blind_to_params,
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
        vec,
        Vec,
        vis,
        visible_private_types,
        volatile,
        warn,
        warn_directory_ownership,
        wasm_import_module,
        wasm_target_feature,
        while_let,
        windows,
        windows_subsystem,
        Yield,
    }
}

#[derive(Copy, Clone, Eq)]
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

    /// Constructs a new identifier with an empty syntax context.
    #[inline]
    pub const fn with_empty_ctxt(name: Symbol) -> Ident {
        Ident::new(name, DUMMY_SP)
    }

    #[inline]
    pub fn invalid() -> Ident {
        Ident::with_empty_ctxt(kw::Invalid)
    }

    /// Maps an interned string to an identifier with an empty syntax context.
    pub fn from_interned_str(string: InternedString) -> Ident {
        Ident::with_empty_ctxt(string.as_symbol())
    }

    /// Maps a string to an identifier with an empty span.
    pub fn from_str(string: &str) -> Ident {
        Ident::with_empty_ctxt(Symbol::intern(string))
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
    /// Identifiers with same string value become same if they came from the same "modern" macro
    /// (e.g., `macro` item, but not `macro_rules` item) and stay different if they came from
    /// different "modern" macros.
    /// Technically, this operation strips all non-opaque marks from ident's syntactic context.
    pub fn modern(self) -> Ident {
        Ident::new(self.name, self.span.modern())
    }

    /// "Normalize" ident for use in comparisons using "local variable hygiene".
    /// Identifiers with same string value become same if they came from the same non-transparent
    /// macro (e.g., `macro` or `macro_rules!` items) and stay different if they came from different
    /// non-transparent macros.
    /// Technically, this operation strips all transparent marks from ident's syntactic context.
    pub fn modern_and_legacy(self) -> Ident {
        Ident::new(self.name, self.span.modern_and_legacy())
    }

    /// Transforms an identifier into one with the same name, but gensymed.
    pub fn gensym(self) -> Ident {
        let name = with_interner(|interner| interner.gensymed(self.name));
        Ident::new(name, self.span)
    }

    /// Transforms an underscore identifier into one with the same name, but
    /// gensymed. Leaves non-underscore identifiers unchanged.
    pub fn gensym_if_underscore(self) -> Ident {
        if self.name == kw::Underscore { self.gensym() } else { self }
    }

    // WARNING: this function is deprecated and will be removed in the future.
    pub fn is_gensymed(self) -> bool {
        with_interner(|interner| interner.is_gensymed(self.name))
    }

    pub fn as_str(self) -> LocalInternedString {
        self.name.as_str()
    }

    pub fn as_interned_str(self) -> InternedString {
        self.name.as_interned_str()
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
        write!(f, "{}{:?}", self.name, self.span.ctxt())
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.name, f)
    }
}

impl Encodable for Ident {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        if self.span.ctxt().modern() == SyntaxContext::empty() {
            s.emit_str(&self.as_str())
        } else { // FIXME(jseyfried): intercrate hygiene
            let mut string = "#".to_owned();
            string.push_str(&self.as_str());
            s.emit_str(&string)
        }
    }
}

impl Decodable for Ident {
    fn decode<D: Decoder>(d: &mut D) -> Result<Ident, D::Error> {
        let string = d.read_str()?;
        Ok(if !string.starts_with('#') {
            Ident::from_str(&string)
        } else { // FIXME(jseyfried): intercrate hygiene
            Ident::from_str(&string[1..]).gensym()
        })
    }
}

/// A symbol is an interned or gensymed string. A gensym is a symbol that is
/// never equal to any other symbol.
///
/// Conceptually, a gensym can be thought of as a normal symbol with an
/// invisible unique suffix. Gensyms are useful when creating new identifiers
/// that must not match any existing identifiers, e.g. during macro expansion
/// and syntax desugaring. Because gensyms should always be identifiers, all
/// gensym operations are on `Ident` rather than `Symbol`. (Indeed, in the
/// future the gensym-ness may be moved from `Symbol` to hygiene data.)
///
/// Examples:
/// ```
/// assert_eq!(Ident::from_str("x"), Ident::from_str("x"))
/// assert_ne!(Ident::from_str("x").gensym(), Ident::from_str("x"))
/// assert_ne!(Ident::from_str("x").gensym(), Ident::from_str("x").gensym())
/// ```
/// Internally, a symbol is implemented as an index, and all operations
/// (including hashing, equality, and ordering) operate on that index. The use
/// of `newtype_index!` means that `Option<Symbol>` only takes up 4 bytes,
/// because `newtype_index!` reserves the last 256 values for tagging purposes.
///
/// Note that `Symbol` cannot directly be a `newtype_index!` because it
/// implements `fmt::Debug`, `Encodable`, and `Decodable` in special ways.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(SymbolIndex);

newtype_index! {
    pub struct SymbolIndex { .. }
}

impl Symbol {
    const fn new(n: u32) -> Self {
        Symbol(SymbolIndex::from_u32_const(n))
    }

    /// Maps a string to its interned representation.
    pub fn intern(string: &str) -> Self {
        with_interner(|interner| interner.intern(string))
    }

    pub fn as_str(self) -> LocalInternedString {
        with_interner(|interner| unsafe {
            LocalInternedString {
                string: std::mem::transmute::<&str, &str>(interner.get(self))
            }
        })
    }

    pub fn as_interned_str(self) -> InternedString {
        with_interner(|interner| InternedString {
            symbol: interner.interned(self)
        })
    }

    pub fn as_u32(self) -> u32 {
        self.0.as_u32()
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let is_gensymed = with_interner(|interner| interner.is_gensymed(*self));
        if is_gensymed {
            write!(f, "{}({:?})", self, self.0)
        } else {
            write!(f, "{}", self)
        }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.as_str(), f)
    }
}

impl Encodable for Symbol {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(&self.as_str())
    }
}

impl Decodable for Symbol {
    fn decode<D: Decoder>(d: &mut D) -> Result<Symbol, D::Error> {
        Ok(Symbol::intern(&d.read_str()?))
    }
}

// The `&'static str`s in this type actually point into the arena.
//
// Note that normal symbols are indexed upward from 0, and gensyms are indexed
// downward from SymbolIndex::MAX_AS_U32.
#[derive(Default)]
pub struct Interner {
    arena: DroplessArena,
    names: FxHashMap<&'static str, Symbol>,
    strings: Vec<&'static str>,
    gensyms: Vec<Symbol>,
}

impl Interner {
    fn prefill(init: &[&'static str]) -> Self {
        Interner {
            strings: init.into(),
            names: init.iter().copied().zip((0..).map(Symbol::new)).collect(),
            ..Default::default()
        }
    }

    pub fn intern(&mut self, string: &str) -> Symbol {
        if let Some(&name) = self.names.get(string) {
            return name;
        }

        let name = Symbol::new(self.strings.len() as u32);

        // `from_utf8_unchecked` is safe since we just allocated a `&str` which is known to be
        // UTF-8.
        let string: &str = unsafe {
            str::from_utf8_unchecked(self.arena.alloc_slice(string.as_bytes()))
        };
        // It is safe to extend the arena allocation to `'static` because we only access
        // these while the arena is still alive.
        let string: &'static str =  unsafe {
            &*(string as *const str)
        };
        self.strings.push(string);
        self.names.insert(string, name);
        name
    }

    fn interned(&self, symbol: Symbol) -> Symbol {
        if (symbol.0.as_usize()) < self.strings.len() {
            symbol
        } else {
            self.gensyms[(SymbolIndex::MAX_AS_U32 - symbol.0.as_u32()) as usize]
        }
    }

    fn gensymed(&mut self, symbol: Symbol) -> Symbol {
        self.gensyms.push(symbol);
        Symbol::new(SymbolIndex::MAX_AS_U32 - self.gensyms.len() as u32 + 1)
    }

    fn is_gensymed(&mut self, symbol: Symbol) -> bool {
        symbol.0.as_usize() >= self.strings.len()
    }

    // Get the symbol as a string. `Symbol::as_str()` should be used in
    // preference to this function.
    pub fn get(&self, symbol: Symbol) -> &str {
        match self.strings.get(symbol.0.as_usize()) {
            Some(string) => string,
            None => {
                let symbol = self.gensyms[(SymbolIndex::MAX_AS_U32 - symbol.0.as_u32()) as usize];
                self.strings[symbol.0.as_usize()]
            }
        }
    }
}

// This module has a very short name because it's used a lot.
pub mod kw {
    use super::Symbol;
    keywords!();
}

// This module has a very short name because it's used a lot.
pub mod sym {
    use std::convert::TryInto;
    use super::Symbol;

    symbols!();

    // Get the symbol for an integer. The first few non-negative integers each
    // have a static symbol and therefore are fast.
    pub fn integer<N: TryInto<usize> + Copy + ToString>(n: N) -> Symbol {
        if let Result::Ok(idx) = n.try_into() {
            if let Option::Some(&sym) = digits_array.get(idx) {
                return sym;
            }
        }
        Symbol::intern(&n.to_string())
    }
}

impl Symbol {
    fn is_used_keyword_2018(self) -> bool {
        self == kw::Dyn
    }

    fn is_unused_keyword_2018(self) -> bool {
        self >= kw::Async && self <= kw::Try
    }

    /// Used for sanity checking rustdoc keyword sections.
    pub fn is_doc_keyword(self) -> bool {
        self <= kw::Union
    }

    /// A keyword or reserved identifier that can be used as a path segment.
    pub fn is_path_segment_keyword(self) -> bool {
        self == kw::Super ||
        self == kw::SelfLower ||
        self == kw::SelfUpper ||
        self == kw::Crate ||
        self == kw::PathRoot ||
        self == kw::DollarCrate
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
        self.name >= kw::As && self.name <= kw::While ||
        self.name.is_used_keyword_2018() && self.span.rust_2018()
    }

    /// Returns `true` if the token is a keyword reserved for possible future use.
    pub fn is_unused_keyword(self) -> bool {
        // Note: `span.edition()` is relatively expensive, don't call it unless necessary.
        self.name >= kw::Abstract && self.name <= kw::Yield ||
        self.name.is_unused_keyword_2018() && self.span.rust_2018()
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

// If an interner exists, return it. Otherwise, prepare a fresh one.
#[inline]
fn with_interner<T, F: FnOnce(&mut Interner) -> T>(f: F) -> T {
    GLOBALS.with(|globals| f(&mut *globals.symbol_interner.lock()))
}

/// An alternative to `Symbol` and `InternedString`, useful when the chars
/// within the symbol need to be accessed. It is best used for temporary
/// values.
///
/// Because the interner outlives any thread which uses this type, we can
/// safely treat `string` which points to interner data, as an immortal string,
/// as long as this type never crosses between threads.
//
// FIXME: ensure that the interner outlives any thread which uses
// `LocalInternedString`, by creating a new thread right after constructing the
// interner.
#[derive(Clone, Copy, Hash, PartialOrd, Eq, Ord)]
pub struct LocalInternedString {
    string: &'static str,
}

impl LocalInternedString {
    /// Maps a string to its interned representation.
    pub fn intern(string: &str) -> Self {
        let string = with_interner(|interner| {
            let symbol = interner.intern(string);
            interner.strings[symbol.0.as_usize()]
        });
        LocalInternedString {
            string: unsafe { std::mem::transmute::<&str, &str>(string) }
        }
    }

    pub fn as_interned_str(self) -> InternedString {
        InternedString {
            symbol: Symbol::intern(self.string)
        }
    }

    #[inline]
    pub fn get(&self) -> &str {
        // This returns a valid string since we ensure that `self` outlives the interner
        // by creating the interner on a thread which outlives threads which can access it.
        // This type cannot move to a thread which outlives the interner since it does
        // not implement Send.
        self.string
    }
}

impl<U: ?Sized> std::convert::AsRef<U> for LocalInternedString
where
    str: std::convert::AsRef<U>
{
    #[inline]
    fn as_ref(&self) -> &U {
        self.string.as_ref()
    }
}

impl<T: std::ops::Deref<Target = str>> std::cmp::PartialEq<T> for LocalInternedString {
    fn eq(&self, other: &T) -> bool {
        self.string == other.deref()
    }
}

impl std::cmp::PartialEq<LocalInternedString> for str {
    fn eq(&self, other: &LocalInternedString) -> bool {
        self == other.string
    }
}

impl<'a> std::cmp::PartialEq<LocalInternedString> for &'a str {
    fn eq(&self, other: &LocalInternedString) -> bool {
        *self == other.string
    }
}

impl std::cmp::PartialEq<LocalInternedString> for String {
    fn eq(&self, other: &LocalInternedString) -> bool {
        self == other.string
    }
}

impl<'a> std::cmp::PartialEq<LocalInternedString> for &'a String {
    fn eq(&self, other: &LocalInternedString) -> bool {
        *self == other.string
    }
}

impl !Send for LocalInternedString {}
impl !Sync for LocalInternedString {}

impl std::ops::Deref for LocalInternedString {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str { self.string }
}

impl fmt::Debug for LocalInternedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.string, f)
    }
}

impl fmt::Display for LocalInternedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.string, f)
    }
}

impl Decodable for LocalInternedString {
    fn decode<D: Decoder>(d: &mut D) -> Result<LocalInternedString, D::Error> {
        Ok(LocalInternedString::intern(&d.read_str()?))
    }
}

impl Encodable for LocalInternedString {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(self.string)
    }
}

/// An alternative to `Symbol` that is focused on string contents. It has two
/// main differences to `Symbol`.
///
/// First, its implementations of `Hash`, `PartialOrd` and `Ord` work with the
/// string chars rather than the symbol integer. This is useful when hash
/// stability is required across compile sessions, or a guaranteed sort
/// ordering is required.
///
/// Second, gensym-ness is irrelevant. E.g.:
/// ```
/// assert_ne!(Symbol::gensym("x"), Symbol::gensym("x"))
/// assert_eq!(Symbol::gensym("x").as_interned_str(), Symbol::gensym("x").as_interned_str())
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct InternedString {
    symbol: Symbol,
}

impl InternedString {
    /// Maps a string to its interned representation.
    pub fn intern(string: &str) -> Self {
        InternedString {
            symbol: Symbol::intern(string)
        }
    }

    pub fn with<F: FnOnce(&str) -> R, R>(self, f: F) -> R {
        let str = with_interner(|interner| {
            interner.get(self.symbol) as *const str
        });
        // This is safe because the interner keeps string alive until it is dropped.
        // We can access it because we know the interner is still alive since we use a
        // scoped thread local to access it, and it was alive at the beginning of this scope
        unsafe { f(&*str) }
    }

    fn with2<F: FnOnce(&str, &str) -> R, R>(self, other: &InternedString, f: F) -> R {
        let (self_str, other_str) = with_interner(|interner| {
            (interner.get(self.symbol) as *const str,
             interner.get(other.symbol) as *const str)
        });
        // This is safe for the same reason that `with` is safe.
        unsafe { f(&*self_str, &*other_str) }
    }

    pub fn as_symbol(self) -> Symbol {
        self.symbol
    }

    pub fn as_str(self) -> LocalInternedString {
        self.symbol.as_str()
    }
}

impl Hash for InternedString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.with(|str| str.hash(state))
    }
}

impl PartialOrd<InternedString> for InternedString {
    fn partial_cmp(&self, other: &InternedString) -> Option<Ordering> {
        if self.symbol == other.symbol {
            return Some(Ordering::Equal);
        }
        self.with2(other, |self_str, other_str| self_str.partial_cmp(other_str))
    }
}

impl Ord for InternedString {
    fn cmp(&self, other: &InternedString) -> Ordering {
        if self.symbol == other.symbol {
            return Ordering::Equal;
        }
        self.with2(other, |self_str, other_str| self_str.cmp(other_str))
    }
}

impl fmt::Debug for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.with(|str| fmt::Debug::fmt(&str, f))
    }
}

impl fmt::Display for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.with(|str| fmt::Display::fmt(&str, f))
    }
}

impl Decodable for InternedString {
    fn decode<D: Decoder>(d: &mut D) -> Result<InternedString, D::Error> {
        Ok(InternedString::intern(&d.read_str()?))
    }
}

impl Encodable for InternedString {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        self.with(|string| s.emit_str(string))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Globals;
    use crate::edition;

    #[test]
    fn interner_tests() {
        let mut i: Interner = Interner::default();
        // first one is zero:
        assert_eq!(i.intern("dog"), Symbol::new(0));
        // re-use gets the same entry:
        assert_eq!(i.intern("dog"), Symbol::new(0));
        // different string gets a different #:
        assert_eq!(i.intern("cat"), Symbol::new(1));
        assert_eq!(i.intern("cat"), Symbol::new(1));
        // dog is still at zero
        assert_eq!(i.intern("dog"), Symbol::new(0));
        let z = i.intern("zebra");
        assert_eq!(i.gensymed(z), Symbol::new(SymbolIndex::MAX_AS_U32));
        // gensym of same string gets new number:
        assert_eq!(i.gensymed(z), Symbol::new(SymbolIndex::MAX_AS_U32 - 1));
        // gensym of *existing* string gets new number:
        let d = i.intern("dog");
        assert_eq!(i.gensymed(d), Symbol::new(SymbolIndex::MAX_AS_U32 - 2));
    }

    #[test]
    fn without_first_quote_test() {
        GLOBALS.set(&Globals::new(edition::DEFAULT_EDITION), || {
            let i = Ident::from_str("'break");
            assert_eq!(i.without_first_quote().name, kw::Break);
        });
    }
}
