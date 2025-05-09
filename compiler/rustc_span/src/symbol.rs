//! An "interner" is a data structure that associates values with usize tags and
//! allows bidirectional lookup; i.e., given a value, one can easily find the
//! type, and vice versa.

use std::hash::{Hash, Hasher};
use std::{fmt, str};

use rustc_arena::DroplessArena;
use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::stable_hasher::{
    HashStable, StableCompare, StableHasher, ToStableHashKey,
};
use rustc_data_structures::sync::Lock;
use rustc_macros::{Decodable, Encodable, HashStable_Generic, symbols};

use crate::{DUMMY_SP, Edition, Span, with_session_globals};

#[cfg(test)]
mod tests;

// The proc macro code for this is in `compiler/rustc_macros/src/symbols.rs`.
symbols! {
    // This list includes things that are definitely keywords (e.g. `if`),
    // a few things that are definitely not keywords (e.g. the empty symbol,
    // `{{root}}`) and things where there is disagreement between people and/or
    // documents (such as the Rust Reference) about whether it is a keyword
    // (e.g. `_`).
    //
    // If you modify this list, adjust any relevant `Symbol::{is,can_be}_*`
    // predicates and `used_keywords`. Also consider adding new keywords to the
    // `ui/parser/raw/raw-idents.rs` test.
    Keywords {
        // Special reserved identifiers used internally for elided lifetimes,
        // unnamed method parameters, crate root module, error recovery etc.
        // Matching predicates: `is_special`/`is_reserved`
        //
        // Notes about `kw::Empty`:
        // - Its use can blur the lines between "empty symbol" and "no symbol".
        //   Using `Option<Symbol>` is preferable, where possible, because that
        //   is unambiguous.
        // - For dummy symbols that are never used and absolutely must be
        //   present, it's better to use `sym::dummy` than `kw::Empty`, because
        //   it's clearer that it's intended as a dummy value, and more likely
        //   to be detected if it accidentally does get used.
        // tidy-alphabetical-start
        DollarCrate:        "$crate",
        Empty:              "",
        PathRoot:           "{{root}}",
        Underscore:         "_",
        // tidy-alphabetical-end

        // Keywords that are used in stable Rust.
        // Matching predicates: `is_used_keyword_always`/`is_reserved`
        // tidy-alphabetical-start
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
        // tidy-alphabetical-end

        // Keywords that are used in unstable Rust or reserved for future use.
        // Matching predicates: `is_unused_keyword_always`/`is_reserved`
        // tidy-alphabetical-start
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
        // tidy-alphabetical-end

        // Edition-specific keywords that are used in stable Rust.
        // Matching predicates: `is_used_keyword_conditional`/`is_reserved` (if
        // the edition suffices)
        // tidy-alphabetical-start
        Async:              "async", // >= 2018 Edition only
        Await:              "await", // >= 2018 Edition only
        Dyn:                "dyn", // >= 2018 Edition only
        // tidy-alphabetical-end

        // Edition-specific keywords that are used in unstable Rust or reserved for future use.
        // Matching predicates: `is_unused_keyword_conditional`/`is_reserved` (if
        // the edition suffices)
        // tidy-alphabetical-start
        Gen:                "gen", // >= 2024 Edition only
        Try:                "try", // >= 2018 Edition only
        // tidy-alphabetical-end

        // "Lifetime keywords": regular keywords with a leading `'`.
        // Matching predicates: none
        // tidy-alphabetical-start
        StaticLifetime:     "'static",
        UnderscoreLifetime: "'_",
        // tidy-alphabetical-end

        // Weak keywords, have special meaning only in specific contexts.
        // Matching predicates: `is_weak`
        // tidy-alphabetical-start
        Auto:               "auto",
        Builtin:            "builtin",
        Catch:              "catch",
        ContractEnsures:    "contract_ensures",
        ContractRequires:   "contract_requires",
        Default:            "default",
        MacroRules:         "macro_rules",
        Raw:                "raw",
        Reuse:              "reuse",
        Safe:               "safe",
        Union:              "union",
        Yeet:               "yeet",
        // tidy-alphabetical-end
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
        Abi,
        AcqRel,
        Acquire,
        Any,
        Arc,
        ArcWeak,
        Argument,
        ArrayIntoIter,
        AsMut,
        AsRef,
        AssertParamIsClone,
        AssertParamIsCopy,
        AssertParamIsEq,
        AsyncGenFinished,
        AsyncGenPending,
        AsyncGenReady,
        AtomicBool,
        AtomicI128,
        AtomicI16,
        AtomicI32,
        AtomicI64,
        AtomicI8,
        AtomicIsize,
        AtomicPtr,
        AtomicU128,
        AtomicU16,
        AtomicU32,
        AtomicU64,
        AtomicU8,
        AtomicUsize,
        BTreeEntry,
        BTreeMap,
        BTreeSet,
        BinaryHeap,
        Borrow,
        BorrowMut,
        Break,
        C,
        CStr,
        C_dash_unwind: "C-unwind",
        CallOnceFuture,
        CallRefFuture,
        Capture,
        Cell,
        Center,
        Child,
        Cleanup,
        Clone,
        CoercePointee,
        CoercePointeeValidated,
        CoerceUnsized,
        Command,
        ConstParamTy,
        ConstParamTy_,
        Context,
        Continue,
        ControlFlow,
        Copy,
        Cow,
        Debug,
        DebugStruct,
        Decodable,
        Decoder,
        Default,
        Deref,
        DiagMessage,
        Diagnostic,
        DirBuilder,
        DispatchFromDyn,
        Display,
        DoubleEndedIterator,
        Duration,
        Encodable,
        Encoder,
        Enumerate,
        Eq,
        Equal,
        Err,
        Error,
        File,
        FileType,
        FmtArgumentsNew,
        Fn,
        FnMut,
        FnOnce,
        Formatter,
        From,
        FromIterator,
        FromResidual,
        FsOpenOptions,
        FsPermissions,
        FusedIterator,
        Future,
        GlobalAlloc,
        Hash,
        HashMap,
        HashMapEntry,
        HashSet,
        Hasher,
        Implied,
        InCleanup,
        IndexOutput,
        Input,
        Instant,
        Into,
        IntoFuture,
        IntoIterator,
        IoBufRead,
        IoLines,
        IoRead,
        IoSeek,
        IoWrite,
        IpAddr,
        IrTyKind,
        Is,
        Item,
        ItemContext,
        IterEmpty,
        IterOnce,
        IterPeekable,
        Iterator,
        IteratorItem,
        Layout,
        Left,
        LinkedList,
        LintDiagnostic,
        LintPass,
        LocalKey,
        Mutex,
        MutexGuard,
        N,
        NonNull,
        NonZero,
        None,
        Normal,
        Ok,
        Option,
        Ord,
        Ordering,
        OsStr,
        OsString,
        Output,
        Param,
        ParamSet,
        PartialEq,
        PartialOrd,
        Path,
        PathBuf,
        Pending,
        PinCoerceUnsized,
        Pointer,
        Poll,
        ProcMacro,
        ProceduralMasqueradeDummyType,
        Range,
        RangeBounds,
        RangeCopy,
        RangeFrom,
        RangeFromCopy,
        RangeFull,
        RangeInclusive,
        RangeInclusiveCopy,
        RangeMax,
        RangeMin,
        RangeSub,
        RangeTo,
        RangeToInclusive,
        Rc,
        RcWeak,
        Ready,
        Receiver,
        RefCell,
        RefCellRef,
        RefCellRefMut,
        Relaxed,
        Release,
        Result,
        ResumeTy,
        Return,
        Right,
        Rust,
        RustaceansAreAwesome,
        RwLock,
        RwLockReadGuard,
        RwLockWriteGuard,
        Saturating,
        SeekFrom,
        SelfTy,
        Send,
        SeqCst,
        Sized,
        SliceIndex,
        SliceIter,
        Some,
        SpanCtxt,
        Stdin,
        String,
        StructuralPartialEq,
        SubdiagMessage,
        Subdiagnostic,
        SymbolIntern,
        Sync,
        SyncUnsafeCell,
        T,
        Target,
        This,
        ToOwned,
        ToString,
        TokenStream,
        Trait,
        Try,
        TryCaptureGeneric,
        TryCapturePrintable,
        TryFrom,
        TryInto,
        Ty,
        TyCtxt,
        TyKind,
        Unknown,
        Unsize,
        UnsizedConstParamTy,
        Upvars,
        Vec,
        VecDeque,
        Waker,
        Wrapper,
        Wrapping,
        Yield,
        _DECLS,
        _Self,
        __D,
        __H,
        __S,
        __awaitee,
        __try_var,
        _d,
        _e,
        _task_context,
        a32,
        aarch64_target_feature,
        aarch64_unstable_target_feature,
        aarch64_ver_target_feature,
        abi,
        abi_amdgpu_kernel,
        abi_avr_interrupt,
        abi_c_cmse_nonsecure_call,
        abi_efiapi,
        abi_gpu_kernel,
        abi_msp430_interrupt,
        abi_ptx,
        abi_riscv_interrupt,
        abi_sysv64,
        abi_thiscall,
        abi_unadjusted,
        abi_vectorcall,
        abi_x86_interrupt,
        abort,
        add,
        add_assign,
        add_with_overflow,
        address,
        adt_const_params,
        advanced_slice_patterns,
        adx_target_feature,
        aes,
        aggregate_raw_ptr,
        alias,
        align,
        alignment,
        all,
        alloc,
        alloc_error_handler,
        alloc_layout,
        alloc_zeroed,
        allocator,
        allocator_api,
        allocator_internals,
        allow,
        allow_fail,
        allow_internal_unsafe,
        allow_internal_unstable,
        altivec,
        alu32,
        always,
        and,
        and_then,
        anon,
        anon_adt,
        anon_assoc,
        anonymous_lifetime_in_impl_trait,
        any,
        append_const_msg,
        apx_target_feature,
        arbitrary_enum_discriminant,
        arbitrary_self_types,
        arbitrary_self_types_pointers,
        areg,
        args,
        arith_offset,
        arm,
        arm_target_feature,
        array,
        as_ptr,
        as_ref,
        as_str,
        asm,
        asm_const,
        asm_experimental_arch,
        asm_experimental_reg,
        asm_goto,
        asm_goto_with_outputs,
        asm_sym,
        asm_unwind,
        assert,
        assert_eq,
        assert_eq_macro,
        assert_inhabited,
        assert_macro,
        assert_mem_uninitialized_valid,
        assert_ne_macro,
        assert_receiver_is_total_eq,
        assert_zero_valid,
        asserting,
        associated_const_equality,
        associated_consts,
        associated_type_bounds,
        associated_type_defaults,
        associated_types,
        assume,
        assume_init,
        asterisk: "*",
        async_await,
        async_call,
        async_call_mut,
        async_call_once,
        async_closure,
        async_drop,
        async_drop_in_place,
        async_fn,
        async_fn_in_dyn_trait,
        async_fn_in_trait,
        async_fn_kind_helper,
        async_fn_kind_upvars,
        async_fn_mut,
        async_fn_once,
        async_fn_once_output,
        async_fn_track_caller,
        async_fn_traits,
        async_for_loop,
        async_iterator,
        async_iterator_poll_next,
        async_trait_bounds,
        atomic,
        atomic_mod,
        atomics,
        att_syntax,
        attr,
        attr_literals,
        attributes,
        audit_that,
        augmented_assignments,
        auto_traits,
        autodiff,
        automatically_derived,
        avx,
        avx10_target_feature,
        avx512_target_feature,
        avx512bw,
        avx512f,
        await_macro,
        bang,
        begin_panic,
        bench,
        bevy_ecs,
        bikeshed_guaranteed_no_drop,
        bin,
        binaryheap_iter,
        bind_by_move_pattern_guards,
        bindings_after_at,
        bitand,
        bitand_assign,
        bitor,
        bitor_assign,
        bitreverse,
        bitxor,
        bitxor_assign,
        black_box,
        block,
        bool,
        bool_then,
        borrowck_graphviz_format,
        borrowck_graphviz_postflow,
        box_new,
        box_patterns,
        box_syntax,
        bpf_target_feature,
        braced_empty_structs,
        branch,
        breakpoint,
        bridge,
        bswap,
        btreemap_contains_key,
        btreemap_insert,
        btreeset_iter,
        builtin_syntax,
        c,
        c_dash_variadic,
        c_str,
        c_str_literals,
        c_unwind,
        c_variadic,
        c_void,
        call,
        call_mut,
        call_once,
        call_once_future,
        call_ref_future,
        caller_location,
        capture_disjoint_fields,
        carrying_mul_add,
        catch_unwind,
        cause,
        cdylib,
        ceilf128,
        ceilf16,
        ceilf32,
        ceilf64,
        cfg,
        cfg_accessible,
        cfg_attr,
        cfg_attr_multi,
        cfg_attr_trace: "<cfg_attr>", // must not be a valid identifier
        cfg_boolean_literals,
        cfg_contract_checks,
        cfg_doctest,
        cfg_emscripten_wasm_eh,
        cfg_eval,
        cfg_fmt_debug,
        cfg_hide,
        cfg_overflow_checks,
        cfg_panic,
        cfg_relocation_model,
        cfg_sanitize,
        cfg_sanitizer_cfi,
        cfg_target_abi,
        cfg_target_compact,
        cfg_target_feature,
        cfg_target_has_atomic,
        cfg_target_has_atomic_equal_alignment,
        cfg_target_has_reliable_f16_f128,
        cfg_target_thread_local,
        cfg_target_vendor,
        cfg_trace: "<cfg>", // must not be a valid identifier
        cfg_ub_checks,
        cfg_version,
        cfi,
        cfi_encoding,
        char,
        char_is_ascii,
        child_id,
        child_kill,
        client,
        clippy,
        clobber_abi,
        clone,
        clone_closures,
        clone_fn,
        clone_from,
        closure,
        closure_lifetime_binder,
        closure_to_fn_coercion,
        closure_track_caller,
        cmp,
        cmp_max,
        cmp_min,
        cmp_ord_max,
        cmp_ord_min,
        cmp_partialeq_eq,
        cmp_partialeq_ne,
        cmp_partialord_cmp,
        cmp_partialord_ge,
        cmp_partialord_gt,
        cmp_partialord_le,
        cmp_partialord_lt,
        cmpxchg16b_target_feature,
        cmse_nonsecure_entry,
        coerce_pointee_validated,
        coerce_unsized,
        cold,
        cold_path,
        collapse_debuginfo,
        column,
        compare_bytes,
        compare_exchange,
        compare_exchange_weak,
        compile_error,
        compiler,
        compiler_builtins,
        compiler_fence,
        concat,
        concat_bytes,
        concat_idents,
        conservative_impl_trait,
        console,
        const_allocate,
        const_async_blocks,
        const_closures,
        const_compare_raw_pointers,
        const_constructor,
        const_deallocate,
        const_destruct,
        const_eval_limit,
        const_eval_select,
        const_evaluatable_checked,
        const_extern_fn,
        const_fn,
        const_fn_floating_point_arithmetic,
        const_fn_fn_ptr_basics,
        const_fn_trait_bound,
        const_fn_transmute,
        const_fn_union,
        const_fn_unsize,
        const_for,
        const_format_args,
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
        const_panic_fmt,
        const_param_ty,
        const_precise_live_drops,
        const_ptr_cast,
        const_raw_ptr_deref,
        const_raw_ptr_to_usize_cast,
        const_refs_to_cell,
        const_refs_to_static,
        const_trait,
        const_trait_bound_opt_out,
        const_trait_impl,
        const_try,
        const_ty_placeholder: "<const_ty>",
        constant,
        constructor,
        contract_build_check_ensures,
        contract_check_ensures,
        contract_check_requires,
        contract_checks,
        contracts,
        contracts_ensures,
        contracts_internals,
        contracts_requires,
        convert_identity,
        copy,
        copy_closures,
        copy_nonoverlapping,
        copysignf128,
        copysignf16,
        copysignf32,
        copysignf64,
        core,
        core_panic,
        core_panic_2015_macro,
        core_panic_2021_macro,
        core_panic_macro,
        coroutine,
        coroutine_clone,
        coroutine_resume,
        coroutine_return,
        coroutine_state,
        coroutine_yield,
        coroutines,
        cosf128,
        cosf16,
        cosf32,
        cosf64,
        count,
        coverage,
        coverage_attribute,
        cr,
        crate_in_paths,
        crate_local,
        crate_name,
        crate_type,
        crate_visibility_modifier,
        crt_dash_static: "crt-static",
        csky_target_feature,
        cstr_type,
        cstring_as_c_str,
        cstring_type,
        ctlz,
        ctlz_nonzero,
        ctpop,
        cttz,
        cttz_nonzero,
        custom_attribute,
        custom_code_classes_in_docs,
        custom_derive,
        custom_inner_attributes,
        custom_mir,
        custom_test_frameworks,
        d,
        d32,
        dbg_macro,
        dead_code,
        dealloc,
        debug,
        debug_assert_eq_macro,
        debug_assert_macro,
        debug_assert_ne_macro,
        debug_assertions,
        debug_struct,
        debug_struct_fields_finish,
        debug_tuple,
        debug_tuple_fields_finish,
        debugger_visualizer,
        decl_macro,
        declare_lint_pass,
        decode,
        default_alloc_error_handler,
        default_field_values,
        default_fn,
        default_lib_allocator,
        default_method_body_is_const,
        // --------------------------
        // Lang items which are used only for experiments with auto traits with default bounds.
        // These lang items are not actually defined in core/std. Experiment is a part of
        // `MCP: Low level components for async drop`(https://github.com/rust-lang/compiler-team/issues/727)
        default_trait1,
        default_trait2,
        default_trait3,
        default_trait4,
        // --------------------------
        default_type_parameter_fallback,
        default_type_params,
        define_opaque,
        delayed_bug_from_inside_query,
        deny,
        deprecated,
        deprecated_safe,
        deprecated_suggestion,
        deref,
        deref_method,
        deref_mut,
        deref_mut_method,
        deref_patterns,
        deref_pure,
        deref_target,
        derive,
        derive_coerce_pointee,
        derive_const,
        derive_default_enum,
        derive_smart_pointer,
        destruct,
        destructuring_assignment,
        diagnostic,
        diagnostic_namespace,
        direct,
        discriminant_kind,
        discriminant_type,
        discriminant_value,
        disjoint_bitor,
        dispatch_from_dyn,
        div,
        div_assign,
        diverging_block_default,
        do_not_recommend,
        doc,
        doc_alias,
        doc_auto_cfg,
        doc_cfg,
        doc_cfg_hide,
        doc_keyword,
        doc_masked,
        doc_notable_trait,
        doc_primitive,
        doc_spotlight,
        doctest,
        document_private_items,
        dotdot: "..",
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
        dummy: "<!dummy!>", // use this instead of `kw::Empty` for symbols that won't be used
        dummy_cgu_name,
        dylib,
        dyn_compatible_for_dispatch,
        dyn_metadata,
        dyn_star,
        dyn_trait,
        dynamic_no_pic: "dynamic-no-pic",
        e,
        edition_panic,
        effects,
        eh_catch_typeinfo,
        eh_personality,
        emit,
        emit_enum,
        emit_enum_variant,
        emit_enum_variant_arg,
        emit_struct,
        emit_struct_field,
        emscripten_wasm_eh,
        enable,
        encode,
        end,
        entry_nops,
        enumerate_method,
        env,
        env_CFG_RELEASE: env!("CFG_RELEASE"),
        eprint_macro,
        eprintln_macro,
        eq,
        ergonomic_clones,
        ermsb_target_feature,
        exact_div,
        except,
        exchange_malloc,
        exclusive_range_pattern,
        exhaustive_integer_patterns,
        exhaustive_patterns,
        existential_type,
        exp2f128,
        exp2f16,
        exp2f32,
        exp2f64,
        expect,
        expected,
        expf128,
        expf16,
        expf32,
        expf64,
        explicit_extern_abis,
        explicit_generic_args_with_impl_trait,
        explicit_tail_calls,
        export_name,
        export_stable,
        expr,
        expr_2021,
        expr_fragment_specifier_2024,
        extended_key_value_attributes,
        extended_varargs_abi_support,
        extern_absolute_paths,
        extern_crate_item_prelude,
        extern_crate_self,
        extern_in_paths,
        extern_prelude,
        extern_system_varargs,
        extern_types,
        external,
        external_doc,
        f,
        f128,
        f128_nan,
        f16,
        f16_nan,
        f16c_target_feature,
        f32,
        f32_epsilon,
        f32_legacy_const_digits,
        f32_legacy_const_epsilon,
        f32_legacy_const_infinity,
        f32_legacy_const_mantissa_dig,
        f32_legacy_const_max,
        f32_legacy_const_max_10_exp,
        f32_legacy_const_max_exp,
        f32_legacy_const_min,
        f32_legacy_const_min_10_exp,
        f32_legacy_const_min_exp,
        f32_legacy_const_min_positive,
        f32_legacy_const_nan,
        f32_legacy_const_neg_infinity,
        f32_legacy_const_radix,
        f32_nan,
        f64,
        f64_epsilon,
        f64_legacy_const_digits,
        f64_legacy_const_epsilon,
        f64_legacy_const_infinity,
        f64_legacy_const_mantissa_dig,
        f64_legacy_const_max,
        f64_legacy_const_max_10_exp,
        f64_legacy_const_max_exp,
        f64_legacy_const_min,
        f64_legacy_const_min_10_exp,
        f64_legacy_const_min_exp,
        f64_legacy_const_min_positive,
        f64_legacy_const_nan,
        f64_legacy_const_neg_infinity,
        f64_legacy_const_radix,
        f64_nan,
        fabsf128,
        fabsf16,
        fabsf32,
        fabsf64,
        fadd_algebraic,
        fadd_fast,
        fake_variadic,
        fallback,
        fdiv_algebraic,
        fdiv_fast,
        feature,
        fence,
        ferris: "🦀",
        fetch_update,
        ffi,
        ffi_const,
        ffi_pure,
        ffi_returns_twice,
        field,
        field_init_shorthand,
        file,
        file_options,
        flags,
        float,
        float_to_int_unchecked,
        floorf128,
        floorf16,
        floorf32,
        floorf64,
        fmaf128,
        fmaf16,
        fmaf32,
        fmaf64,
        fmt,
        fmt_debug,
        fmul_algebraic,
        fmul_fast,
        fmuladdf128,
        fmuladdf16,
        fmuladdf32,
        fmuladdf64,
        fn_align,
        fn_body,
        fn_delegation,
        fn_must_use,
        fn_mut,
        fn_once,
        fn_once_output,
        fn_ptr_addr,
        fn_ptr_trait,
        forbid,
        forget,
        format,
        format_args,
        format_args_capture,
        format_args_macro,
        format_args_nl,
        format_argument,
        format_arguments,
        format_count,
        format_macro,
        format_placeholder,
        format_unsafe_arg,
        freeze,
        freeze_impls,
        freg,
        frem_algebraic,
        frem_fast,
        from,
        from_desugaring,
        from_fn,
        from_iter,
        from_iter_fn,
        from_output,
        from_residual,
        from_size_align_unchecked,
        from_str_method,
        from_u16,
        from_usize,
        from_yeet,
        frontmatter,
        fs_create_dir,
        fsub_algebraic,
        fsub_fast,
        fsxr,
        full,
        fundamental,
        fused_iterator,
        future,
        future_drop_poll,
        future_output,
        future_trait,
        gdb_script_file,
        ge,
        gen_blocks,
        gen_future,
        generator_clone,
        generators,
        generic_arg_infer,
        generic_assert,
        generic_associated_types,
        generic_associated_types_extended,
        generic_const_exprs,
        generic_const_items,
        generic_const_parameter_types,
        generic_param_attrs,
        generic_pattern_types,
        get_context,
        global_alloc_ty,
        global_allocator,
        global_asm,
        global_registration,
        globs,
        gt,
        guard_patterns,
        half_open_range_patterns,
        half_open_range_patterns_in_slices,
        hash,
        hashmap_contains_key,
        hashmap_drain_ty,
        hashmap_insert,
        hashmap_iter_mut_ty,
        hashmap_iter_ty,
        hashmap_keys_ty,
        hashmap_values_mut_ty,
        hashmap_values_ty,
        hashset_drain_ty,
        hashset_iter,
        hashset_iter_ty,
        hexagon_target_feature,
        hidden,
        hint,
        homogeneous_aggregate,
        host,
        html_favicon_url,
        html_logo_url,
        html_no_source,
        html_playground_url,
        html_root_url,
        hwaddress,
        i,
        i128,
        i128_legacy_const_max,
        i128_legacy_const_min,
        i128_legacy_fn_max_value,
        i128_legacy_fn_min_value,
        i128_legacy_mod,
        i128_type,
        i16,
        i16_legacy_const_max,
        i16_legacy_const_min,
        i16_legacy_fn_max_value,
        i16_legacy_fn_min_value,
        i16_legacy_mod,
        i32,
        i32_legacy_const_max,
        i32_legacy_const_min,
        i32_legacy_fn_max_value,
        i32_legacy_fn_min_value,
        i32_legacy_mod,
        i64,
        i64_legacy_const_max,
        i64_legacy_const_min,
        i64_legacy_fn_max_value,
        i64_legacy_fn_min_value,
        i64_legacy_mod,
        i8,
        i8_legacy_const_max,
        i8_legacy_const_min,
        i8_legacy_fn_max_value,
        i8_legacy_fn_min_value,
        i8_legacy_mod,
        ident,
        if_let,
        if_let_guard,
        if_let_rescope,
        if_while_or_patterns,
        ignore,
        impl_header_lifetime_elision,
        impl_lint_pass,
        impl_trait_in_assoc_type,
        impl_trait_in_bindings,
        impl_trait_in_fn_trait_return,
        impl_trait_projections,
        implement_via_object,
        implied_by,
        import,
        import_name_type,
        import_shadowing,
        import_trait_associated_functions,
        imported_main,
        in_band_lifetimes,
        include,
        include_bytes,
        include_bytes_macro,
        include_str,
        include_str_macro,
        inclusive_range_syntax,
        index,
        index_mut,
        infer_outlives_requirements,
        infer_static_outlives_requirements,
        inherent_associated_types,
        inherit,
        inlateout,
        inline,
        inline_const,
        inline_const_pat,
        inout,
        instant_now,
        instruction_set,
        integer_: "integer", // underscore to avoid clashing with the function `sym::integer` below
        integral,
        internal_features,
        into_async_iter_into_iter,
        into_future,
        into_iter,
        intra_doc_pointers,
        intrinsics,
        intrinsics_unaligned_volatile_load,
        intrinsics_unaligned_volatile_store,
        io_stderr,
        io_stdout,
        irrefutable_let_patterns,
        is,
        is_val_statically_known,
        isa_attribute,
        isize,
        isize_legacy_const_max,
        isize_legacy_const_min,
        isize_legacy_fn_max_value,
        isize_legacy_fn_min_value,
        isize_legacy_mod,
        issue,
        issue_5723_bootstrap,
        issue_tracker_base_url,
        item,
        item_like_imports,
        iter,
        iter_cloned,
        iter_copied,
        iter_filter,
        iter_mut,
        iter_repeat,
        iterator,
        iterator_collect_fn,
        kcfi,
        keylocker_x86,
        keyword,
        kind,
        kreg,
        kreg0,
        label,
        label_break_value,
        lahfsahf_target_feature,
        lang,
        lang_items,
        large_assignments,
        lateout,
        lazy_normalization_consts,
        lazy_type_alias,
        le,
        legacy_receiver,
        len,
        let_chains,
        let_else,
        lhs,
        lib,
        libc,
        lifetime,
        lifetime_capture_rules_2024,
        lifetimes,
        likely,
        line,
        link,
        link_arg_attribute,
        link_args,
        link_cfg,
        link_llvm_intrinsics,
        link_name,
        link_ordinal,
        link_section,
        linkage,
        linker,
        linker_messages,
        lint_reasons,
        literal,
        load,
        loaded_from_disk,
        local,
        local_inner_macros,
        log10f128,
        log10f16,
        log10f32,
        log10f64,
        log2f128,
        log2f16,
        log2f32,
        log2f64,
        log_syntax,
        logf128,
        logf16,
        logf32,
        logf64,
        loongarch_target_feature,
        loop_break_value,
        lt,
        m68k_target_feature,
        macro_at_most_once_rep,
        macro_attributes_in_derive_output,
        macro_escape,
        macro_export,
        macro_lifetime_matcher,
        macro_literal_matcher,
        macro_metavar_expr,
        macro_metavar_expr_concat,
        macro_reexport,
        macro_use,
        macro_vis_matcher,
        macros_in_extern,
        main,
        managed_boxes,
        manually_drop,
        map,
        map_err,
        marker,
        marker_trait_attr,
        masked,
        match_beginning_vert,
        match_default_bindings,
        matches_macro,
        maxnumf128,
        maxnumf16,
        maxnumf32,
        maxnumf64,
        may_dangle,
        may_unwind,
        maybe_uninit,
        maybe_uninit_uninit,
        maybe_uninit_zeroed,
        mem_discriminant,
        mem_drop,
        mem_forget,
        mem_replace,
        mem_size_of,
        mem_size_of_val,
        mem_swap,
        mem_uninitialized,
        mem_variant_count,
        mem_zeroed,
        member_constraints,
        memory,
        memtag,
        message,
        meta,
        metadata_type,
        min_align_of,
        min_align_of_val,
        min_const_fn,
        min_const_generics,
        min_const_unsafe_fn,
        min_exhaustive_patterns,
        min_generic_const_args,
        min_specialization,
        min_type_alias_impl_trait,
        minnumf128,
        minnumf16,
        minnumf32,
        minnumf64,
        mips_target_feature,
        mir_assume,
        mir_basic_block,
        mir_call,
        mir_cast_ptr_to_ptr,
        mir_cast_transmute,
        mir_checked,
        mir_copy_for_deref,
        mir_debuginfo,
        mir_deinit,
        mir_discriminant,
        mir_drop,
        mir_field,
        mir_goto,
        mir_len,
        mir_make_place,
        mir_move,
        mir_offset,
        mir_ptr_metadata,
        mir_retag,
        mir_return,
        mir_return_to,
        mir_set_discriminant,
        mir_static,
        mir_static_mut,
        mir_storage_dead,
        mir_storage_live,
        mir_tail_call,
        mir_unreachable,
        mir_unwind_cleanup,
        mir_unwind_continue,
        mir_unwind_resume,
        mir_unwind_terminate,
        mir_unwind_terminate_reason,
        mir_unwind_unreachable,
        mir_variant,
        miri,
        mmx_reg,
        modifiers,
        module,
        module_path,
        more_maybe_bounds,
        more_qualified_paths,
        more_struct_aliases,
        movbe_target_feature,
        move_ref_pattern,
        move_size_limit,
        movrs_target_feature,
        mul,
        mul_assign,
        mul_with_overflow,
        multiple_supertrait_upcastable,
        must_not_suspend,
        must_use,
        mut_preserve_binding_mode_2024,
        mut_ref,
        naked,
        naked_asm,
        naked_functions,
        naked_functions_rustic_abi,
        naked_functions_target_feature,
        name,
        names,
        native_link_modifiers,
        native_link_modifiers_as_needed,
        native_link_modifiers_bundle,
        native_link_modifiers_verbatim,
        native_link_modifiers_whole_archive,
        natvis_file,
        ne,
        needs_allocator,
        needs_drop,
        needs_panic_runtime,
        neg,
        negate_unsigned,
        negative_bounds,
        negative_impls,
        neon,
        nested,
        never,
        never_patterns,
        never_type,
        never_type_fallback,
        new,
        new_binary,
        new_const,
        new_debug,
        new_debug_noop,
        new_display,
        new_lower_exp,
        new_lower_hex,
        new_octal,
        new_pointer,
        new_range,
        new_unchecked,
        new_upper_exp,
        new_upper_hex,
        new_v1,
        new_v1_formatted,
        next,
        niko,
        nll,
        no,
        no_builtins,
        no_core,
        no_coverage,
        no_crate_inject,
        no_debug,
        no_default_passes,
        no_implicit_prelude,
        no_inline,
        no_link,
        no_main,
        no_mangle,
        no_sanitize,
        no_stack_check,
        no_std,
        nomem,
        non_ascii_idents,
        non_exhaustive,
        non_exhaustive_omitted_patterns_lint,
        non_lifetime_binders,
        non_modrs_mods,
        none,
        nontemporal_store,
        noop_method_borrow,
        noop_method_clone,
        noop_method_deref,
        noreturn,
        nostack,
        not,
        notable_trait,
        note,
        object_safe_for_dispatch,
        of,
        off,
        offset,
        offset_of,
        offset_of_enum,
        offset_of_nested,
        offset_of_slice,
        ok_or_else,
        omit_gdb_pretty_printer_section,
        on,
        on_unimplemented,
        opaque,
        opaque_module_name_placeholder: "<opaque>",
        open_options_new,
        ops,
        opt_out_copy,
        optimize,
        optimize_attribute,
        optin_builtin_traits,
        option,
        option_env,
        option_expect,
        option_unwrap,
        options,
        or,
        or_patterns,
        ord_cmp_method,
        os_str_to_os_string,
        os_string_as_os_str,
        other,
        out,
        overflow_checks,
        overlapping_marker_traits,
        owned_box,
        packed,
        packed_bundled_libs,
        panic,
        panic_2015,
        panic_2021,
        panic_abort,
        panic_any,
        panic_bounds_check,
        panic_cannot_unwind,
        panic_const_add_overflow,
        panic_const_async_fn_resumed,
        panic_const_async_fn_resumed_drop,
        panic_const_async_fn_resumed_panic,
        panic_const_async_gen_fn_resumed,
        panic_const_async_gen_fn_resumed_drop,
        panic_const_async_gen_fn_resumed_panic,
        panic_const_coroutine_resumed,
        panic_const_coroutine_resumed_drop,
        panic_const_coroutine_resumed_panic,
        panic_const_div_by_zero,
        panic_const_div_overflow,
        panic_const_gen_fn_none,
        panic_const_gen_fn_none_drop,
        panic_const_gen_fn_none_panic,
        panic_const_mul_overflow,
        panic_const_neg_overflow,
        panic_const_rem_by_zero,
        panic_const_rem_overflow,
        panic_const_shl_overflow,
        panic_const_shr_overflow,
        panic_const_sub_overflow,
        panic_fmt,
        panic_handler,
        panic_impl,
        panic_implementation,
        panic_in_cleanup,
        panic_info,
        panic_location,
        panic_misaligned_pointer_dereference,
        panic_nounwind,
        panic_null_pointer_dereference,
        panic_runtime,
        panic_str_2015,
        panic_unwind,
        panicking,
        param_attrs,
        parent_label,
        partial_cmp,
        partial_ord,
        passes,
        pat,
        pat_param,
        patchable_function_entry,
        path,
        path_main_separator,
        path_to_pathbuf,
        pathbuf_as_path,
        pattern_complexity_limit,
        pattern_parentheses,
        pattern_type,
        pattern_type_range_trait,
        pattern_types,
        permissions_from_mode,
        phantom_data,
        pic,
        pie,
        pin,
        pin_ergonomics,
        platform_intrinsics,
        plugin,
        plugin_registrar,
        plugins,
        pointee,
        pointee_trait,
        pointer,
        pointer_like,
        poll,
        poll_next,
        position,
        post_dash_lto: "post-lto",
        postfix_match,
        powerpc_target_feature,
        powf128,
        powf16,
        powf32,
        powf64,
        powif128,
        powif16,
        powif32,
        powif64,
        pre_dash_lto: "pre-lto",
        precise_capturing,
        precise_capturing_in_traits,
        precise_pointer_size_matching,
        precision,
        pref_align_of,
        prefetch_read_data,
        prefetch_read_instruction,
        prefetch_write_data,
        prefetch_write_instruction,
        prefix_nops,
        preg,
        prelude,
        prelude_import,
        preserves_flags,
        prfchw_target_feature,
        print_macro,
        println_macro,
        proc_dash_macro: "proc-macro",
        proc_macro,
        proc_macro_attribute,
        proc_macro_derive,
        proc_macro_expr,
        proc_macro_gen,
        proc_macro_hygiene,
        proc_macro_internals,
        proc_macro_mod,
        proc_macro_non_items,
        proc_macro_path_invoc,
        process_abort,
        process_exit,
        profiler_builtins,
        profiler_runtime,
        ptr,
        ptr_cast,
        ptr_cast_const,
        ptr_cast_mut,
        ptr_const_is_null,
        ptr_copy,
        ptr_copy_nonoverlapping,
        ptr_eq,
        ptr_from_ref,
        ptr_guaranteed_cmp,
        ptr_is_null,
        ptr_mask,
        ptr_metadata,
        ptr_null,
        ptr_null_mut,
        ptr_offset_from,
        ptr_offset_from_unsigned,
        ptr_read,
        ptr_read_unaligned,
        ptr_read_volatile,
        ptr_replace,
        ptr_slice_from_raw_parts,
        ptr_slice_from_raw_parts_mut,
        ptr_swap,
        ptr_swap_nonoverlapping,
        ptr_unique,
        ptr_write,
        ptr_write_bytes,
        ptr_write_unaligned,
        ptr_write_volatile,
        pub_macro_rules,
        pub_restricted,
        public,
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
        raw_dylib_elf,
        raw_eq,
        raw_identifiers,
        raw_ref_op,
        re_rebalance_coherence,
        read_enum,
        read_enum_variant,
        read_enum_variant_arg,
        read_struct,
        read_struct_field,
        read_via_copy,
        readonly,
        realloc,
        reason,
        receiver,
        receiver_target,
        recursion_limit,
        reexport_test_harness_main,
        ref_pat_eat_one_layer_2024,
        ref_pat_eat_one_layer_2024_structural,
        ref_pat_everywhere,
        ref_unwind_safe_trait,
        reference,
        reflect,
        reg,
        reg16,
        reg32,
        reg64,
        reg_abcd,
        reg_addr,
        reg_byte,
        reg_data,
        reg_iw,
        reg_nonzero,
        reg_pair,
        reg_ptr,
        reg_upper,
        register_attr,
        register_tool,
        relaxed_adts,
        relaxed_struct_unsize,
        relocation_model,
        rem,
        rem_assign,
        repr,
        repr128,
        repr_align,
        repr_align_enum,
        repr_packed,
        repr_simd,
        repr_transparent,
        require,
        reserve_x18: "reserve-x18",
        residual,
        result,
        result_ffi_guarantees,
        result_ok_method,
        resume,
        return_position_impl_trait_in_trait,
        return_type_notation,
        rhs,
        riscv_target_feature,
        rlib,
        ropi,
        ropi_rwpi: "ropi-rwpi",
        rotate_left,
        rotate_right,
        round_ties_even_f128,
        round_ties_even_f16,
        round_ties_even_f32,
        round_ties_even_f64,
        roundf128,
        roundf16,
        roundf32,
        roundf64,
        rt,
        rtm_target_feature,
        rust,
        rust_2015,
        rust_2018,
        rust_2018_preview,
        rust_2021,
        rust_2024,
        rust_analyzer,
        rust_begin_unwind,
        rust_cold_cc,
        rust_eh_catch_typeinfo,
        rust_eh_personality,
        rust_future,
        rust_logo,
        rust_out,
        rustc,
        rustc_abi,
        rustc_allocator,
        rustc_allocator_zeroed,
        rustc_allow_const_fn_unstable,
        rustc_allow_incoherent_impl,
        rustc_allowed_through_unstable_modules,
        rustc_as_ptr,
        rustc_attrs,
        rustc_autodiff,
        rustc_builtin_macro,
        rustc_capture_analysis,
        rustc_clean,
        rustc_coherence_is_core,
        rustc_coinductive,
        rustc_confusables,
        rustc_const_panic_str,
        rustc_const_stable,
        rustc_const_stable_indirect,
        rustc_const_unstable,
        rustc_conversion_suggestion,
        rustc_deallocator,
        rustc_def_path,
        rustc_default_body_unstable,
        rustc_delayed_bug_from_inside_query,
        rustc_deny_explicit_impl,
        rustc_deprecated_safe_2024,
        rustc_diagnostic_item,
        rustc_diagnostic_macros,
        rustc_dirty,
        rustc_do_not_const_check,
        rustc_do_not_implement_via_object,
        rustc_doc_primitive,
        rustc_driver,
        rustc_dummy,
        rustc_dump_def_parents,
        rustc_dump_item_bounds,
        rustc_dump_predicates,
        rustc_dump_user_args,
        rustc_dump_vtable,
        rustc_effective_visibility,
        rustc_evaluate_where_clauses,
        rustc_expected_cgu_reuse,
        rustc_force_inline,
        rustc_has_incoherent_inherent_impls,
        rustc_hidden_type_of_opaques,
        rustc_if_this_changed,
        rustc_inherit_overflow_checks,
        rustc_insignificant_dtor,
        rustc_intrinsic,
        rustc_intrinsic_const_stable_indirect,
        rustc_layout,
        rustc_layout_scalar_valid_range_end,
        rustc_layout_scalar_valid_range_start,
        rustc_legacy_const_generics,
        rustc_lint_diagnostics,
        rustc_lint_opt_deny_field_access,
        rustc_lint_opt_ty,
        rustc_lint_query_instability,
        rustc_lint_untracked_query_information,
        rustc_macro_transparency,
        rustc_main,
        rustc_mir,
        rustc_must_implement_one_of,
        rustc_never_returns_null_ptr,
        rustc_never_type_options,
        rustc_no_implicit_autorefs,
        rustc_no_mir_inline,
        rustc_nonnull_optimization_guaranteed,
        rustc_nounwind,
        rustc_object_lifetime_default,
        rustc_on_unimplemented,
        rustc_outlives,
        rustc_paren_sugar,
        rustc_partition_codegened,
        rustc_partition_reused,
        rustc_pass_by_value,
        rustc_peek,
        rustc_peek_liveness,
        rustc_peek_maybe_init,
        rustc_peek_maybe_uninit,
        rustc_preserve_ub_checks,
        rustc_private,
        rustc_proc_macro_decls,
        rustc_promotable,
        rustc_pub_transparent,
        rustc_reallocator,
        rustc_regions,
        rustc_reservation_impl,
        rustc_serialize,
        rustc_skip_during_method_dispatch,
        rustc_specialization_trait,
        rustc_std_internal_symbol,
        rustc_strict_coherence,
        rustc_symbol_name,
        rustc_test_marker,
        rustc_then_this_would_need,
        rustc_trivial_field_reads,
        rustc_unsafe_specialization_marker,
        rustc_variance,
        rustc_variance_of_opaques,
        rustdoc,
        rustdoc_internals,
        rustdoc_missing_doc_code_examples,
        rustfmt,
        rvalue_static_promotion,
        rwpi,
        s,
        s390x_target_feature,
        safety,
        sanitize,
        sanitizer_cfi_generalize_pointers,
        sanitizer_cfi_normalize_integers,
        sanitizer_runtime,
        saturating_add,
        saturating_div,
        saturating_sub,
        sdylib,
        search_unbox,
        select_unpredictable,
        self_in_typedefs,
        self_struct_ctor,
        semiopaque,
        semitransparent,
        sha2,
        sha3,
        sha512_sm_x86,
        shadow_call_stack,
        shallow,
        shl,
        shl_assign,
        shorter_tail_lifetimes,
        should_panic,
        shr,
        shr_assign,
        sig_dfl,
        sig_ign,
        simd,
        simd_add,
        simd_and,
        simd_arith_offset,
        simd_as,
        simd_bitmask,
        simd_bitreverse,
        simd_bswap,
        simd_cast,
        simd_cast_ptr,
        simd_ceil,
        simd_ctlz,
        simd_ctpop,
        simd_cttz,
        simd_div,
        simd_eq,
        simd_expose_provenance,
        simd_extract,
        simd_extract_dyn,
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
        simd_fsin,
        simd_fsqrt,
        simd_gather,
        simd_ge,
        simd_gt,
        simd_insert,
        simd_insert_dyn,
        simd_le,
        simd_lt,
        simd_masked_load,
        simd_masked_store,
        simd_mul,
        simd_ne,
        simd_neg,
        simd_or,
        simd_reduce_add_ordered,
        simd_reduce_add_unordered,
        simd_reduce_all,
        simd_reduce_and,
        simd_reduce_any,
        simd_reduce_max,
        simd_reduce_min,
        simd_reduce_mul_ordered,
        simd_reduce_mul_unordered,
        simd_reduce_or,
        simd_reduce_xor,
        simd_relaxed_fma,
        simd_rem,
        simd_round,
        simd_saturating_add,
        simd_saturating_sub,
        simd_scatter,
        simd_select,
        simd_select_bitmask,
        simd_shl,
        simd_shr,
        simd_shuffle,
        simd_shuffle_const_generic,
        simd_sub,
        simd_trunc,
        simd_with_exposed_provenance,
        simd_xor,
        since,
        sinf128,
        sinf16,
        sinf32,
        sinf64,
        size,
        size_of,
        size_of_val,
        sized,
        skip,
        slice,
        slice_from_raw_parts,
        slice_from_raw_parts_mut,
        slice_into_vec,
        slice_iter,
        slice_len_fn,
        slice_patterns,
        slicing_syntax,
        soft,
        sparc_target_feature,
        specialization,
        speed,
        spotlight,
        sqrtf128,
        sqrtf16,
        sqrtf32,
        sqrtf64,
        sreg,
        sreg_low16,
        sse,
        sse2,
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
        std_panic,
        std_panic_2015_macro,
        std_panic_macro,
        stmt,
        stmt_expr_attributes,
        stop_after_dataflow,
        store,
        str,
        str_chars,
        str_ends_with,
        str_from_utf8,
        str_from_utf8_mut,
        str_from_utf8_unchecked,
        str_from_utf8_unchecked_mut,
        str_inherent_from_utf8,
        str_inherent_from_utf8_mut,
        str_inherent_from_utf8_unchecked,
        str_inherent_from_utf8_unchecked_mut,
        str_len,
        str_split_whitespace,
        str_starts_with,
        str_trim,
        str_trim_end,
        str_trim_start,
        strict_provenance_lints,
        string_as_mut_str,
        string_as_str,
        string_deref_patterns,
        string_from_utf8,
        string_insert_str,
        string_new,
        string_push_str,
        stringify,
        struct_field_attributes,
        struct_inherit,
        struct_variant,
        structural_match,
        structural_peq,
        sub,
        sub_assign,
        sub_with_overflow,
        suggestion,
        super_let,
        supertrait_item_shadowing,
        sym,
        sync,
        synthetic,
        t32,
        target,
        target_abi,
        target_arch,
        target_endian,
        target_env,
        target_family,
        target_feature,
        target_feature_11,
        target_has_atomic,
        target_has_atomic_equal_alignment,
        target_has_atomic_load_store,
        target_has_reliable_f128,
        target_has_reliable_f128_math,
        target_has_reliable_f16,
        target_has_reliable_f16_math,
        target_os,
        target_pointer_width,
        target_thread_local,
        target_vendor,
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
        test_unstable_lint,
        thread,
        thread_local,
        thread_local_macro,
        three_way_compare,
        thumb2,
        thumb_mode: "thumb-mode",
        tmm_reg,
        to_owned_method,
        to_string,
        to_string_method,
        to_vec,
        todo_macro,
        tool_attributes,
        tool_lints,
        trace_macros,
        track_caller,
        trait_alias,
        trait_upcasting,
        transmute,
        transmute_generic_consts,
        transmute_opts,
        transmute_trait,
        transmute_unchecked,
        transparent,
        transparent_enums,
        transparent_unions,
        trivial_bounds,
        truncf128,
        truncf16,
        truncf32,
        truncf64,
        try_blocks,
        try_capture,
        try_from,
        try_from_fn,
        try_into,
        try_trait_v2,
        tt,
        tuple,
        tuple_indexing,
        tuple_trait,
        two_phase,
        ty,
        type_alias_enum_variants,
        type_alias_impl_trait,
        type_ascribe,
        type_ascription,
        type_changing_struct_update,
        type_const,
        type_id,
        type_ir_infer_ctxt_like,
        type_ir_inherent,
        type_ir_interner,
        type_length_limit,
        type_macros,
        type_name,
        type_privacy_lints,
        typed_swap_nonoverlapping,
        u128,
        u128_legacy_const_max,
        u128_legacy_const_min,
        u128_legacy_fn_max_value,
        u128_legacy_fn_min_value,
        u128_legacy_mod,
        u16,
        u16_legacy_const_max,
        u16_legacy_const_min,
        u16_legacy_fn_max_value,
        u16_legacy_fn_min_value,
        u16_legacy_mod,
        u32,
        u32_legacy_const_max,
        u32_legacy_const_min,
        u32_legacy_fn_max_value,
        u32_legacy_fn_min_value,
        u32_legacy_mod,
        u64,
        u64_legacy_const_max,
        u64_legacy_const_min,
        u64_legacy_fn_max_value,
        u64_legacy_fn_min_value,
        u64_legacy_mod,
        u8,
        u8_legacy_const_max,
        u8_legacy_const_min,
        u8_legacy_fn_max_value,
        u8_legacy_fn_min_value,
        u8_legacy_mod,
        ub_checks,
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
        unimplemented_macro,
        unit,
        universal_impl_trait,
        unix,
        unlikely,
        unmarked_api,
        unnamed_fields,
        unpin,
        unqualified_local_imports,
        unreachable,
        unreachable_2015,
        unreachable_2015_macro,
        unreachable_2021,
        unreachable_code,
        unreachable_display,
        unreachable_macro,
        unrestricted_attribute_tokens,
        unsafe_attributes,
        unsafe_binders,
        unsafe_block_in_unsafe_fn,
        unsafe_cell,
        unsafe_cell_raw_get,
        unsafe_extern_blocks,
        unsafe_fields,
        unsafe_no_drop_flag,
        unsafe_pinned,
        unsafe_unpin,
        unsize,
        unsized_const_param_ty,
        unsized_const_params,
        unsized_fn_params,
        unsized_locals,
        unsized_tuple_coercion,
        unstable,
        unstable_location_reason_default: "this crate is being loaded from the sysroot, an \
                          unstable location; did you mean to load this crate \
                          from crates.io via `Cargo.toml` instead?",
        untagged_unions,
        unused_imports,
        unwind,
        unwind_attributes,
        unwind_safe_trait,
        unwrap,
        unwrap_binder,
        unwrap_or,
        use_cloned,
        use_extern_macros,
        use_nested_groups,
        used,
        used_with_arg,
        using,
        usize,
        usize_legacy_const_max,
        usize_legacy_const_min,
        usize_legacy_fn_max_value,
        usize_legacy_fn_min_value,
        usize_legacy_mod,
        v8plus,
        va_arg,
        va_copy,
        va_end,
        va_list,
        va_start,
        val,
        validity,
        values,
        var,
        variant_count,
        vec,
        vec_as_mut_slice,
        vec_as_slice,
        vec_from_elem,
        vec_is_empty,
        vec_macro,
        vec_new,
        vec_pop,
        vec_reserve,
        vec_with_capacity,
        vecdeque_iter,
        vecdeque_reserve,
        vector,
        version,
        vfp2,
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
        vsx,
        vtable_align,
        vtable_size,
        warn,
        wasip2,
        wasm_abi,
        wasm_import_module,
        wasm_target_feature,
        where_clause_attrs,
        while_let,
        width,
        windows,
        windows_subsystem,
        with_negative_coherence,
        wrap_binder,
        wrapping_add,
        wrapping_div,
        wrapping_mul,
        wrapping_rem,
        wrapping_rem_euclid,
        wrapping_sub,
        wreg,
        write_bytes,
        write_fmt,
        write_macro,
        write_str,
        write_via_move,
        writeln_macro,
        x86_amx_intrinsics,
        x87_reg,
        x87_target_feature,
        xer,
        xmm_reg,
        xop_target_feature,
        yeet_desugar_details,
        yeet_expr,
        yes,
        yield_expr,
        ymm_reg,
        yreg,
        zfh,
        zfhmin,
        zmm_reg,
    }
}

/// Symbols for crates that are part of the stable standard library: `std`, `core`, `alloc`, and
/// `proc_macro`.
pub const STDLIB_STABLE_CRATES: &[Symbol] = &[sym::std, sym::core, sym::alloc, sym::proc_macro];

#[derive(Copy, Clone, Eq, HashStable_Generic, Encodable, Decodable)]
pub struct Ident {
    // `name` should never be the empty symbol. If you are considering that,
    // you are probably conflating "empty identifer with "no identifier" and
    // you should use `Option<Ident>` instead.
    pub name: Symbol,
    pub span: Span,
}

impl Ident {
    #[inline]
    /// Constructs a new identifier from a symbol and a span.
    pub fn new(name: Symbol, span: Span) -> Ident {
        debug_assert_ne!(name, kw::Empty);
        Ident { name, span }
    }

    /// Constructs a new identifier with a dummy span.
    #[inline]
    pub fn with_dummy_span(name: Symbol) -> Ident {
        Ident::new(name, DUMMY_SP)
    }

    // For dummy identifiers that are never used and absolutely must be
    // present. Note that this does *not* use the empty symbol; `sym::dummy`
    // makes it clear that it's intended as a dummy value, and is more likely
    // to be detected if it accidentally does get used.
    #[inline]
    pub fn dummy() -> Ident {
        Ident::with_dummy_span(sym::dummy)
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
    #[inline]
    pub fn normalize_to_macro_rules(self) -> Ident {
        Ident::new(self.name, self.span.normalize_to_macro_rules())
    }

    /// Access the underlying string. This is a slowish operation because it
    /// requires locking the symbol interner.
    ///
    /// Note that the lifetime of the return value is a lie. See
    /// `Symbol::as_str()` for details.
    pub fn as_str(&self) -> &str {
        self.name.as_str()
    }
}

impl PartialEq for Ident {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name && self.span.eq_ctxt(rhs.span)
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

/// The most general type to print identifiers.
///
/// AST pretty-printer is used as a fallback for turning AST structures into token streams for
/// proc macros. Additionally, proc macros may stringify their input and expect it survive the
/// stringification (especially true for proc macro derives written between Rust 1.15 and 1.30).
/// So we need to somehow pretty-print `$crate` in a way preserving at least some of its
/// hygiene data, most importantly name of the crate it refers to.
/// As a result we print `$crate` as `crate` if it refers to the local crate
/// and as `::other_crate_name` if it refers to some other crate.
/// Note, that this is only done if the ident token is printed from inside of AST pretty-printing,
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
    #[inline]
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
    #[orderable]
    struct SymbolIndex {}
}

impl Symbol {
    pub const fn new(n: u32) -> Self {
        Symbol(SymbolIndex::from_u32(n))
    }

    /// Maps a string to its interned representation.
    #[rustc_diagnostic_item = "SymbolIntern"]
    pub fn intern(string: &str) -> Self {
        with_session_globals(|session_globals| session_globals.symbol_interner.intern(string))
    }

    /// Access the underlying string. This is a slowish operation because it
    /// requires locking the symbol interner.
    ///
    /// Note that the lifetime of the return value is a lie. It's not the same
    /// as `&self`, but actually tied to the lifetime of the underlying
    /// interner. Interners are long-lived, and there are very few of them, and
    /// this function is typically used for short-lived things, so in practice
    /// it works out ok.
    pub fn as_str(&self) -> &str {
        with_session_globals(|session_globals| unsafe {
            std::mem::transmute::<&str, &str>(session_globals.symbol_interner.get(*self))
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
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl<CTX> HashStable<CTX> for Symbol {
    #[inline]
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        self.as_str().hash_stable(hcx, hasher);
    }
}

impl<CTX> ToStableHashKey<CTX> for Symbol {
    type KeyType = String;
    #[inline]
    fn to_stable_hash_key(&self, _: &CTX) -> String {
        self.as_str().to_string()
    }
}

impl StableCompare for Symbol {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    fn stable_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

pub(crate) struct Interner(Lock<InternerInner>);

// The `&'static str`s in this type actually point into the arena.
//
// This type is private to prevent accidentally constructing more than one
// `Interner` on the same thread, which makes it easy to mix up `Symbol`s
// between `Interner`s.
struct InternerInner {
    arena: DroplessArena,
    strings: FxIndexSet<&'static str>,
}

impl Interner {
    fn prefill(init: &[&'static str], extra: &[&'static str]) -> Self {
        let strings = FxIndexSet::from_iter(init.iter().copied().chain(extra.iter().copied()));
        assert_eq!(
            strings.len(),
            init.len() + extra.len(),
            "`init` or `extra` contain duplicate symbols",
        );
        Interner(Lock::new(InternerInner { arena: Default::default(), strings }))
    }

    #[inline]
    fn intern(&self, string: &str) -> Symbol {
        let mut inner = self.0.lock();
        if let Some(idx) = inner.strings.get_index_of(string) {
            return Symbol::new(idx as u32);
        }

        let string: &str = inner.arena.alloc_str(string);

        // SAFETY: we can extend the arena allocation to `'static` because we
        // only access these while the arena is still alive.
        let string: &'static str = unsafe { &*(string as *const str) };

        // This second hash table lookup can be avoided by using `RawEntryMut`,
        // but this code path isn't hot enough for it to be worth it. See
        // #91445 for details.
        let (idx, is_new) = inner.strings.insert_full(string);
        debug_assert!(is_new); // due to the get_index_of check above

        Symbol::new(idx as u32)
    }

    /// Get the symbol as a string.
    ///
    /// [`Symbol::as_str()`] should be used in preference to this function.
    fn get(&self, symbol: Symbol) -> &str {
        self.0.lock().strings.get_index(symbol.0.as_usize()).unwrap()
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
    // Used from a macro in `librustc_feature/accepted.rs`
    use super::Symbol;
    pub use super::kw::MacroRules as macro_rules;
    #[doc(inline)]
    pub use super::sym_generated::*;

    /// Get the symbol for an integer.
    ///
    /// The first few non-negative integers each have a static symbol and therefore
    /// are fast.
    pub fn integer<N: TryInto<usize> + Copy + itoa::Integer>(n: N) -> Symbol {
        if let Result::Ok(idx) = n.try_into() {
            if idx < 10 {
                return Symbol::new(super::SYMBOL_DIGITS_BASE + idx as u32);
            }
        }
        let mut buffer = itoa::Buffer::new();
        let printed = buffer.format(n);
        Symbol::intern(printed)
    }
}

impl Symbol {
    fn is_special(self) -> bool {
        self <= kw::Underscore
    }

    fn is_used_keyword_always(self) -> bool {
        self >= kw::As && self <= kw::While
    }

    fn is_unused_keyword_always(self) -> bool {
        self >= kw::Abstract && self <= kw::Yield
    }

    fn is_used_keyword_conditional(self, edition: impl FnOnce() -> Edition) -> bool {
        (self >= kw::Async && self <= kw::Dyn) && edition() >= Edition::Edition2018
    }

    fn is_unused_keyword_conditional(self, edition: impl Copy + FnOnce() -> Edition) -> bool {
        self == kw::Gen && edition().at_least_rust_2024()
            || self == kw::Try && edition().at_least_rust_2018()
    }

    pub fn is_reserved(self, edition: impl Copy + FnOnce() -> Edition) -> bool {
        self.is_special()
            || self.is_used_keyword_always()
            || self.is_unused_keyword_always()
            || self.is_used_keyword_conditional(edition)
            || self.is_unused_keyword_conditional(edition)
    }

    pub fn is_weak(self) -> bool {
        self >= kw::Auto && self <= kw::Yeet
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

    /// Was this symbol predefined in the compiler's `symbols!` macro
    pub fn is_predefined(self) -> bool {
        self.as_u32() < PREDEFINED_SYMBOLS_COUNT
    }
}

impl Ident {
    /// Returns `true` for reserved identifiers used internally for elided lifetimes,
    /// unnamed method parameters, crate root module, error recovery etc.
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

    /// Whether this would be the identifier for a tuple field like `self.0`, as
    /// opposed to a named field like `self.thing`.
    pub fn is_numeric(self) -> bool {
        !self.name.is_empty() && self.as_str().bytes().all(|b| b.is_ascii_digit())
    }
}

/// Collect all the keywords in a given edition into a vector.
///
/// *Note:* Please update this if a new keyword is added beyond the current
/// range.
pub fn used_keywords(edition: impl Copy + FnOnce() -> Edition) -> Vec<Symbol> {
    (kw::DollarCrate.as_u32()..kw::Yeet.as_u32())
        .filter_map(|kw| {
            let kw = Symbol::new(kw);
            if kw.is_used_keyword_always() || kw.is_used_keyword_conditional(edition) {
                Some(kw)
            } else {
                None
            }
        })
        .collect()
}
