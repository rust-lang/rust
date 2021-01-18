//! An "interner" is a data structure that associates values with usize tags and
//! allows bidirectional lookup; i.e., given a value, one can easily find the
//! type, and vice versa.

use rustc_arena::DroplessArena;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};
use rustc_macros::HashStable_Generic;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

use std::cmp::{Ord, PartialEq, PartialOrd};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str;

use crate::{Edition, Span, DUMMY_SP, SESSION_GLOBALS};

#[cfg(test)]
mod tests;

// The proc macro code for this is in `src/librustc_macros/src/symbols.rs`.
symbols! {
    // After modifying this list adjust `is_special`, `is_used_keyword`/`is_unused_keyword`,
    // this should be rarely necessary though if the keywords are kept in alphabetic order.
    Keywords {
        // Special reserved identifiers used internally for elided lifetimes,
        // unnamed method parameters, crate root module, error recovery etc.
        Empty:              "",
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

    // Pre-interned symbols that can be referred to with `rustc_span::sym::*`.
    //
    // The symbol is the stringified identifier unless otherwise specified, in
    // which case the name should mention the non-identifier punctuation.
    // E.g. `sym::proc_dash_macro` represents "proc-macro", and it shouldn't be
    // called `sym::proc_macro` because then it's easy to mistakenly think it
    // represents "proc_macro".
    //
    // As well as the symbols listed, there are symbols for the strings
    // "0", "1", ..., "9", which are accessible via `sym::integer`.
    //
    // The proc macro will abort if symbols are not in alphabetical order (as
    // defined by `impl Ord for str`) or if any symbols are duplicated. Vim
    // users can sort the list by selecting it and executing the command
    // `:'<,'>!LC_ALL=C sort`.
    //
    // There is currently no checking that all symbols are used; that would be
    // nice to have.
    Symbols {
        Alignment,
        Arc,
        Argument,
        ArgumentV1,
        Arguments,
        C,
        CString,
        Center,
        Clone,
        Copy,
        Count,
        Debug,
        Decodable,
        Decoder,
        Default,
        Encodable,
        Encoder,
        Eq,
        Equal,
        Err,
        Error,
        FormatSpec,
        Formatter,
        From,
        Future,
        FxHashMap,
        FxHashSet,
        GlobalAlloc,
        Hash,
        HashMap,
        HashSet,
        Hasher,
        Implied,
        Input,
        IntoIterator,
        Is,
        ItemContext,
        Iterator,
        Layout,
        Left,
        LintPass,
        None,
        Ok,
        Option,
        Ord,
        Ordering,
        Output,
        Param,
        PartialEq,
        PartialOrd,
        Pending,
        Pin,
        Poll,
        ProcMacro,
        ProcMacroHack,
        ProceduralMasqueradeDummyType,
        Range,
        RangeFrom,
        RangeFull,
        RangeInclusive,
        RangeTo,
        RangeToInclusive,
        Rc,
        Ready,
        Result,
        Return,
        Right,
        RustcDecodable,
        RustcEncodable,
        Send,
        Some,
        StructuralEq,
        StructuralPartialEq,
        Sync,
        Target,
        Try,
        Ty,
        TyCtxt,
        TyKind,
        Unknown,
        Vec,
        Yield,
        _DECLS,
        _Self,
        __D,
        __H,
        __S,
        __next,
        __try_var,
        _d,
        _e,
        _task_context,
        a32,
        aarch64_target_feature,
        abi,
        abi_amdgpu_kernel,
        abi_avr_interrupt,
        abi_efiapi,
        abi_msp430_interrupt,
        abi_ptx,
        abi_sysv64,
        abi_thiscall,
        abi_unadjusted,
        abi_vectorcall,
        abi_x86_interrupt,
        abort,
        aborts,
        add,
        add_assign,
        add_with_overflow,
        address,
        advanced_slice_patterns,
        adx_target_feature,
        alias,
        align,
        align_offset,
        alignstack,
        all,
        alloc,
        alloc_error_handler,
        alloc_layout,
        alloc_zeroed,
        allocator,
        allocator_internals,
        allow,
        allow_fail,
        allow_internal_unsafe,
        allow_internal_unstable,
        allow_internal_unstable_backcompat_hack,
        allowed,
        always,
        and,
        and_then,
        any,
        arbitrary_enum_discriminant,
        arbitrary_self_types,
        arith_offset,
        arm,
        arm_target_feature,
        array,
        arrays,
        as_ptr,
        as_str,
        asm,
        assert,
        assert_inhabited,
        assert_macro,
        assert_receiver_is_total_eq,
        assert_uninit_valid,
        assert_zero_valid,
        associated_consts,
        associated_type_bounds,
        associated_type_defaults,
        associated_types,
        assume,
        assume_init,
        async_await,
        async_closure,
        atomics,
        att_syntax,
        attr,
        attr_literals,
        attributes,
        augmented_assignments,
        auto_traits,
        automatically_derived,
        avx512_target_feature,
        await_macro,
        bang,
        begin_panic,
        bench,
        bin,
        bind_by_move_pattern_guards,
        bindings_after_at,
        bitand,
        bitand_assign,
        bitor,
        bitor_assign,
        bitreverse,
        bitxor,
        bitxor_assign,
        block,
        bool,
        borrowck_graphviz_format,
        borrowck_graphviz_postflow,
        borrowck_graphviz_preflow,
        box_free,
        box_patterns,
        box_syntax,
        braced_empty_structs,
        breakpoint,
        bridge,
        bswap,
        c_str,
        c_variadic,
        call,
        call_mut,
        call_once,
        caller_location,
        capture_disjoint_fields,
        cdylib,
        ceilf32,
        ceilf64,
        cfg,
        cfg_accessible,
        cfg_attr,
        cfg_attr_multi,
        cfg_doctest,
        cfg_panic,
        cfg_sanitize,
        cfg_target_feature,
        cfg_target_has_atomic,
        cfg_target_thread_local,
        cfg_target_vendor,
        cfg_version,
        char,
        client,
        clippy,
        clone,
        clone_closures,
        clone_from,
        closure,
        closure_to_fn_coercion,
        cmp,
        cmpxchg16b_target_feature,
        cmse_nonsecure_entry,
        coerce_unsized,
        cold,
        column,
        compile_error,
        compiler_builtins,
        concat,
        concat_idents,
        conservative_impl_trait,
        console,
        const_allocate,
        const_compare_raw_pointers,
        const_constructor,
        const_eval_limit,
        const_evaluatable_checked,
        const_extern_fn,
        const_fn,
        const_fn_floating_point_arithmetic,
        const_fn_fn_ptr_basics,
        const_fn_transmute,
        const_fn_union,
        const_generics,
        const_generics_defaults,
        const_if_match,
        const_impl_trait,
        const_in_array_repeat_expressions,
        const_indexing,
        const_let,
        const_loop,
        const_mut_refs,
        const_panic,
        const_precise_live_drops,
        const_ptr,
        const_raw_ptr_deref,
        const_raw_ptr_to_usize_cast,
        const_refs_to_cell,
        const_slice_ptr,
        const_trait_bound_opt_out,
        const_trait_impl,
        const_transmute,
        constant,
        constructor,
        contents,
        context,
        convert,
        copy,
        copy_closures,
        copy_nonoverlapping,
        copysignf32,
        copysignf64,
        core,
        core_intrinsics,
        core_panic_macro,
        cosf32,
        cosf64,
        crate_id,
        crate_in_paths,
        crate_local,
        crate_name,
        crate_type,
        crate_visibility_modifier,
        crt_dash_static: "crt-static",
        cstring_type,
        ctlz,
        ctlz_nonzero,
        ctpop,
        cttz,
        cttz_nonzero,
        custom_attribute,
        custom_derive,
        custom_inner_attributes,
        custom_test_frameworks,
        d,
        dead_code,
        dealloc,
        debug,
        debug_assert_macro,
        debug_assertions,
        debug_struct,
        debug_trait,
        debug_trait_builder,
        debug_tuple,
        decl_macro,
        declare_lint_pass,
        decode,
        default_alloc_error_handler,
        default_lib_allocator,
        default_type_parameter_fallback,
        default_type_params,
        delay_span_bug_from_inside_query,
        deny,
        deprecated,
        deref,
        deref_method,
        deref_mut,
        deref_target,
        derive,
        destructuring_assignment,
        diagnostic,
        direct,
        discriminant_kind,
        discriminant_type,
        discriminant_value,
        dispatch_from_dyn,
        div,
        div_assign,
        doc,
        doc_alias,
        doc_cfg,
        doc_keyword,
        doc_masked,
        doc_spotlight,
        doctest,
        document_private_items,
        dotdot_in_tuple_patterns,
        dotdoteq_in_patterns,
        dreg,
        dreg_low16,
        dreg_low8,
        drop,
        drop_in_place,
        drop_types_in_const,
        dropck_eyepatch,
        dropck_parametricity,
        dylib,
        dyn_trait,
        edition_macro_pats,
        eh_catch_typeinfo,
        eh_personality,
        emit_enum,
        emit_enum_variant,
        emit_enum_variant_arg,
        emit_struct,
        emit_struct_field,
        enable,
        enclosing_scope,
        encode,
        env,
        eq,
        ermsb_target_feature,
        err,
        exact_div,
        except,
        exchange_malloc,
        exclusive_range_pattern,
        exhaustive_integer_patterns,
        exhaustive_patterns,
        existential_type,
        exp2f32,
        exp2f64,
        expect,
        expected,
        expf32,
        expf64,
        export_name,
        expr,
        extended_key_value_attributes,
        extern_absolute_paths,
        extern_crate_item_prelude,
        extern_crate_self,
        extern_in_paths,
        extern_prelude,
        extern_types,
        external_doc,
        f,
        f16c_target_feature,
        f32,
        f32_runtime,
        f64,
        f64_runtime,
        fabsf32,
        fabsf64,
        fadd_fast,
        fdiv_fast,
        feature,
        ffi,
        ffi_const,
        ffi_pure,
        ffi_returns_twice,
        field,
        field_init_shorthand,
        file,
        fill,
        finish,
        flags,
        float_to_int_unchecked,
        floorf32,
        floorf64,
        fmaf32,
        fmaf64,
        fmt,
        fmt_internals,
        fmul_fast,
        fn_must_use,
        fn_mut,
        fn_once,
        fn_once_output,
        forbid,
        forget,
        format,
        format_args,
        format_args_capture,
        format_args_nl,
        freeze,
        freg,
        frem_fast,
        from,
        from_desugaring,
        from_error,
        from_generator,
        from_method,
        from_ok,
        from_size_align_unchecked,
        from_trait,
        from_usize,
        fsub_fast,
        fundamental,
        future,
        future_trait,
        ge,
        gen_future,
        gen_kill,
        generator,
        generator_state,
        generators,
        generic_associated_types,
        generic_param_attrs,
        get_context,
        global_allocator,
        global_asm,
        globs,
        gt,
        half_open_range_patterns,
        hash,
        hexagon_target_feature,
        hidden,
        homogeneous_aggregate,
        html_favicon_url,
        html_logo_url,
        html_no_source,
        html_playground_url,
        html_root_url,
        i,
        i128,
        i128_type,
        i16,
        i32,
        i64,
        i8,
        ident,
        if_let,
        if_let_guard,
        if_while_or_patterns,
        ignore,
        impl_header_lifetime_elision,
        impl_lint_pass,
        impl_macros,
        impl_trait_in_bindings,
        import_shadowing,
        in_band_lifetimes,
        include,
        include_bytes,
        include_str,
        inclusive_range_syntax,
        index,
        index_mut,
        infer_outlives_requirements,
        infer_static_outlives_requirements,
        inlateout,
        inline,
        inline_const,
        inout,
        instruction_set,
        intel,
        into_iter,
        into_result,
        intra_doc_pointers,
        intrinsics,
        irrefutable_let_patterns,
        isa_attribute,
        isize,
        issue,
        issue_5723_bootstrap,
        issue_tracker_base_url,
        item,
        item_like_imports,
        iter,
        keyword,
        kind,
        kreg,
        label,
        label_break_value,
        lang,
        lang_items,
        lateout,
        lazy_normalization_consts,
        le,
        let_chains,
        lhs,
        lib,
        libc,
        lifetime,
        likely,
        line,
        link,
        link_args,
        link_cfg,
        link_llvm_intrinsics,
        link_name,
        link_ordinal,
        link_section,
        linkage,
        lint_reasons,
        literal,
        llvm_asm,
        local,
        local_inner_macros,
        log10f32,
        log10f64,
        log2f32,
        log2f64,
        log_syntax,
        logf32,
        logf64,
        loop_break_value,
        lt,
        macro_at_most_once_rep,
        macro_escape,
        macro_export,
        macro_lifetime_matcher,
        macro_literal_matcher,
        macro_reexport,
        macro_use,
        macro_vis_matcher,
        macros_in_extern,
        main,
        managed_boxes,
        manually_drop,
        map,
        marker,
        marker_trait_attr,
        masked,
        match_beginning_vert,
        match_default_bindings,
        maxnumf32,
        maxnumf64,
        may_dangle,
        maybe_uninit,
        maybe_uninit_uninit,
        maybe_uninit_zeroed,
        mem_uninitialized,
        mem_zeroed,
        member_constraints,
        memory,
        message,
        meta,
        min_align_of,
        min_align_of_val,
        min_const_fn,
        min_const_generics,
        min_const_unsafe_fn,
        min_specialization,
        minnumf32,
        minnumf64,
        mips_target_feature,
        misc,
        module,
        module_path,
        more_struct_aliases,
        movbe_target_feature,
        move_ref_pattern,
        mul,
        mul_assign,
        mul_with_overflow,
        must_use,
        mut_ptr,
        mut_slice_ptr,
        naked,
        naked_functions,
        name,
        ne,
        nearbyintf32,
        nearbyintf64,
        needs_allocator,
        needs_drop,
        needs_panic_runtime,
        neg,
        negate_unsigned,
        negative_impls,
        never,
        never_type,
        never_type_fallback,
        new,
        new_unchecked,
        next,
        nll,
        no,
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
        no_niche,
        no_sanitize,
        no_stack_check,
        no_start,
        no_std,
        nomem,
        non_ascii_idents,
        non_exhaustive,
        non_modrs_mods,
        none_error,
        nontemporal_store,
        nontrapping_dash_fptoint: "nontrapping-fptoint",
        noreturn,
        nostack,
        not,
        note,
        object_safe_for_dispatch,
        of,
        offset,
        omit_gdb_pretty_printer_section,
        on,
        on_unimplemented,
        oom,
        opaque,
        ops,
        opt_out_copy,
        optimize,
        optimize_attribute,
        optin_builtin_traits,
        option,
        option_env,
        option_type,
        options,
        or,
        or_patterns,
        other,
        out,
        overlapping_marker_traits,
        owned_box,
        packed,
        panic,
        panic_abort,
        panic_bounds_check,
        panic_handler,
        panic_impl,
        panic_implementation,
        panic_info,
        panic_location,
        panic_runtime,
        panic_str,
        panic_unwind,
        panicking,
        param_attrs,
        parent_trait,
        partial_cmp,
        partial_ord,
        passes,
        pat,
        pat2018,
        pat2021,
        path,
        pattern_parentheses,
        phantom_data,
        pin,
        pinned,
        platform_intrinsics,
        plugin,
        plugin_registrar,
        plugins,
        pointer,
        pointer_trait,
        pointer_trait_fmt,
        poll,
        position,
        post_dash_lto: "post-lto",
        powerpc_target_feature,
        powf32,
        powf64,
        powif32,
        powif64,
        pre_dash_lto: "pre-lto",
        precise_pointer_size_matching,
        precision,
        pref_align_of,
        prefetch_read_data,
        prefetch_read_instruction,
        prefetch_write_data,
        prefetch_write_instruction,
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
        profiler_builtins,
        profiler_runtime,
        ptr_guaranteed_eq,
        ptr_guaranteed_ne,
        ptr_offset_from,
        pub_restricted,
        pure,
        pushpop_unsafe,
        qreg,
        qreg_low4,
        qreg_low8,
        quad_precision_float,
        question_mark,
        quote,
        range_inclusive_new,
        raw_dylib,
        raw_identifiers,
        raw_ref_op,
        re_rebalance_coherence,
        read_enum,
        read_enum_variant,
        read_enum_variant_arg,
        read_struct,
        read_struct_field,
        readonly,
        realloc,
        reason,
        receiver,
        recursion_limit,
        reexport_test_harness_main,
        reference,
        reflect,
        reg,
        reg16,
        reg32,
        reg64,
        reg_abcd,
        reg_byte,
        reg_thumb,
        register_attr,
        register_tool,
        relaxed_adts,
        rem,
        rem_assign,
        repr,
        repr128,
        repr_align,
        repr_align_enum,
        repr_no_niche,
        repr_packed,
        repr_simd,
        repr_transparent,
        result,
        result_type,
        rhs,
        rintf32,
        rintf64,
        riscv_target_feature,
        rlib,
        rotate_left,
        rotate_right,
        roundf32,
        roundf64,
        rt,
        rtm_target_feature,
        rust,
        rust_2015_preview,
        rust_2018_preview,
        rust_2021_preview,
        rust_begin_unwind,
        rust_eh_catch_typeinfo,
        rust_eh_personality,
        rust_eh_register_frames,
        rust_eh_unregister_frames,
        rust_oom,
        rustc,
        rustc_allocator,
        rustc_allocator_nounwind,
        rustc_allow_const_fn_unstable,
        rustc_args_required_const,
        rustc_attrs,
        rustc_builtin_macro,
        rustc_capture_analysis,
        rustc_clean,
        rustc_const_stable,
        rustc_const_unstable,
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
        rustc_peek_indirectly_mutable,
        rustc_peek_liveness,
        rustc_peek_maybe_init,
        rustc_peek_maybe_uninit,
        rustc_polymorphize_error,
        rustc_private,
        rustc_proc_macro_decls,
        rustc_promotable,
        rustc_regions,
        rustc_reservation_impl,
        rustc_serialize,
        rustc_specialization_trait,
        rustc_stable,
        rustc_std_internal_symbol,
        rustc_symbol_name,
        rustc_synthetic,
        rustc_test_marker,
        rustc_then_this_would_need,
        rustc_unsafe_specialization_marker,
        rustc_variance,
        rustfmt,
        rvalue_static_promotion,
        sanitize,
        sanitizer_runtime,
        saturating_add,
        saturating_sub,
        self_in_typedefs,
        self_struct_ctor,
        semitransparent,
        send_trait,
        shl,
        shl_assign,
        should_panic,
        shr,
        shr_assign,
        simd,
        simd_add,
        simd_and,
        simd_bitmask,
        simd_cast,
        simd_ceil,
        simd_div,
        simd_eq,
        simd_extract,
        simd_fabs,
        simd_fcos,
        simd_fexp,
        simd_fexp2,
        simd_ffi,
        simd_flog,
        simd_flog10,
        simd_flog2,
        simd_floor,
        simd_fma,
        simd_fmax,
        simd_fmin,
        simd_fpow,
        simd_fpowi,
        simd_fsin,
        simd_fsqrt,
        simd_gather,
        simd_ge,
        simd_gt,
        simd_insert,
        simd_le,
        simd_lt,
        simd_mul,
        simd_ne,
        simd_or,
        simd_reduce_add_ordered,
        simd_reduce_add_unordered,
        simd_reduce_all,
        simd_reduce_and,
        simd_reduce_any,
        simd_reduce_max,
        simd_reduce_max_nanless,
        simd_reduce_min,
        simd_reduce_min_nanless,
        simd_reduce_mul_ordered,
        simd_reduce_mul_unordered,
        simd_reduce_or,
        simd_reduce_xor,
        simd_rem,
        simd_saturating_add,
        simd_saturating_sub,
        simd_scatter,
        simd_select,
        simd_select_bitmask,
        simd_shl,
        simd_shr,
        simd_sub,
        simd_xor,
        since,
        sinf32,
        sinf64,
        size,
        size_of,
        size_of_val,
        sized,
        slice,
        slice_alloc,
        slice_patterns,
        slice_u8,
        slice_u8_alloc,
        slicing_syntax,
        soft,
        specialization,
        speed,
        spotlight,
        sqrtf32,
        sqrtf64,
        sreg,
        sreg_low16,
        sse4a_target_feature,
        stable,
        staged_api,
        start,
        state,
        static_in_const,
        static_nobundle,
        static_recursion,
        staticlib,
        std,
        std_inject,
        std_panic_macro,
        stmt,
        stmt_expr_attributes,
        stop_after_dataflow,
        str,
        str_alloc,
        string_type,
        stringify,
        struct_field_attributes,
        struct_inherit,
        struct_variant,
        structural_match,
        structural_peq,
        structural_teq,
        sty,
        sub,
        sub_assign,
        sub_with_overflow,
        suggestion,
        sym,
        sync,
        sync_trait,
        t32,
        target_arch,
        target_endian,
        target_env,
        target_family,
        target_feature,
        target_feature_11,
        target_has_atomic,
        target_has_atomic_equal_alignment,
        target_has_atomic_load_store,
        target_os,
        target_pointer_width,
        target_target_vendor,
        target_thread_local,
        target_vendor,
        task,
        tbm_target_feature,
        termination,
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
        truncf32,
        truncf64,
        try_blocks,
        try_trait,
        tt,
        tuple,
        tuple_from_req,
        tuple_indexing,
        two_phase,
        ty,
        type_alias_enum_variants,
        type_alias_impl_trait,
        type_ascription,
        type_id,
        type_length_limit,
        type_macros,
        type_name,
        u128,
        u16,
        u32,
        u64,
        u8,
        unaligned_volatile_load,
        unaligned_volatile_store,
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
        unit,
        universal_impl_trait,
        unix,
        unlikely,
        unmarked_api,
        unpin,
        unreachable,
        unreachable_code,
        unrestricted_attribute_tokens,
        unsafe_block_in_unsafe_fn,
        unsafe_cell,
        unsafe_no_drop_flag,
        unsize,
        unsized_fn_params,
        unsized_locals,
        unsized_tuple_coercion,
        unstable,
        untagged_unions,
        unused_qualifications,
        unwind,
        unwind_attributes,
        unwrap,
        unwrap_or,
        use_extern_macros,
        use_nested_groups,
        used,
        usize,
        v1,
        va_arg,
        va_copy,
        va_end,
        va_list,
        va_start,
        val,
        var,
        variant_count,
        vec,
        vec_type,
        version,
        vis,
        visible_private_types,
        volatile,
        volatile_copy_memory,
        volatile_copy_nonoverlapping_memory,
        volatile_load,
        volatile_set_memory,
        volatile_store,
        vreg,
        vreg_low16,
        warn,
        wasm_import_module,
        wasm_target_feature,
        while_let,
        width,
        windows,
        windows_subsystem,
        wrapping_add,
        wrapping_mul,
        wrapping_sub,
        write_bytes,
        xmm_reg,
        ymm_reg,
        zmm_reg,
    }
}

#[derive(Copy, Clone, Eq, HashStable_Generic, Encodable, Decodable)]
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
        Ident::with_dummy_span(kw::Empty)
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
        } else if self.symbol == kw::DollarCrate {
            if let Some(span) = self.convert_dollar_crate {
                let converted = span.ctxt().dollar_crate_name();
                if !converted.is_path_segment_keyword() {
                    f.write_str("::")?;
                }
                return fmt::Display::fmt(&converted, f);
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

    pub fn is_empty(self) -> bool {
        self == kw::Empty
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
        fmt::Debug::fmt(&self.as_str(), f)
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.as_str(), f)
    }
}

impl<S: Encoder> Encodable<S> for Symbol {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(&self.as_str())
    }
}

impl<D: Decoder> Decodable<D> for Symbol {
    #[inline]
    fn decode(d: &mut D) -> Result<Symbol, D::Error> {
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
//
// The `FxHashMap`+`Vec` pair could be replaced by `FxIndexSet`, but #75278
// found that to regress performance up to 2% in some cases. This might be
// revisited after further improvements to `indexmap`.
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
    pub use super::kw_generated::*;
}

// This module has a very short name because it's used a lot.
/// This module contains all the defined non-keyword `Symbol`s.
///
/// Given that `sym` is imported, use them like `sym::symbol_name`.
/// For example `sym::rustfmt` or `sym::u8`.
pub mod sym {
    use super::Symbol;
    use std::convert::TryInto;

    pub use super::sym_generated::*;

    // Used from a macro in `librustc_feature/accepted.rs`
    pub use super::kw::MacroRules as macro_rules;

    /// Get the symbol for an integer.
    ///
    /// The first few non-negative integers each have a static symbol and therefore
    /// are fast.
    pub fn integer<N: TryInto<usize> + Copy + ToString>(n: N) -> Symbol {
        if let Result::Ok(idx) = n.try_into() {
            if idx < 10 {
                return Symbol::new(super::SYMBOL_DIGITS_BASE + idx as u32);
            }
        }
        Symbol::intern(&n.to_string())
    }
}

impl Symbol {
    fn is_special(self) -> bool {
        self <= kw::Underscore
    }

    fn is_used_keyword_always(self) -> bool {
        self >= kw::As && self <= kw::While
    }

    fn is_used_keyword_conditional(self, edition: impl FnOnce() -> Edition) -> bool {
        (self >= kw::Async && self <= kw::Dyn) && edition() >= Edition::Edition2018
    }

    fn is_unused_keyword_always(self) -> bool {
        self >= kw::Abstract && self <= kw::Yield
    }

    fn is_unused_keyword_conditional(self, edition: impl FnOnce() -> Edition) -> bool {
        self == kw::Try && edition() >= Edition::Edition2018
    }

    pub fn is_reserved(self, edition: impl Copy + FnOnce() -> Edition) -> bool {
        self.is_special()
            || self.is_used_keyword_always()
            || self.is_unused_keyword_always()
            || self.is_used_keyword_conditional(edition)
            || self.is_unused_keyword_conditional(edition)
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

    /// Returns `true` if this symbol can be a raw identifier.
    pub fn can_be_raw(self) -> bool {
        self != kw::Empty && self != kw::Underscore && !self.is_path_segment_keyword()
    }
}

impl Ident {
    // Returns `true` for reserved identifiers used internally for elided lifetimes,
    // unnamed method parameters, crate root module, error recovery etc.
    pub fn is_special(self) -> bool {
        self.name.is_special()
    }

    /// Returns `true` if the token is a keyword used in the language.
    pub fn is_used_keyword(self) -> bool {
        // Note: `span.edition()` is relatively expensive, don't call it unless necessary.
        self.name.is_used_keyword_always()
            || self.name.is_used_keyword_conditional(|| self.span.edition())
    }

    /// Returns `true` if the token is a keyword reserved for possible future use.
    pub fn is_unused_keyword(self) -> bool {
        // Note: `span.edition()` is relatively expensive, don't call it unless necessary.
        self.name.is_unused_keyword_always()
            || self.name.is_unused_keyword_conditional(|| self.span.edition())
    }

    /// Returns `true` if the token is either a special identifier or a keyword.
    pub fn is_reserved(self) -> bool {
        // Note: `span.edition()` is relatively expensive, don't call it unless necessary.
        self.name.is_reserved(|| self.span.edition())
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
    SESSION_GLOBALS.with(|session_globals| f(&mut *session_globals.symbol_interner.lock()))
}

/// An alternative to [`Symbol`], useful when the chars within the symbol need to
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
/// - `&*ss` is a `&str` (and `match &*ss { ... }` is a common pattern).
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
