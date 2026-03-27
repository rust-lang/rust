options! {
    UnstableOptions, UnstableOptionsTargetModifiers, Z_OPTIONS, dbopts, "Z", "unstable",

    // If you add a new option, please update:
    // - compiler/rustc_interface/src/tests.rs
    // - src/doc/unstable-book/src/compiler-flags

    // tidy-alphabetical-start
    allow_features: Option<Vec<String>> = (None, parse_opt_comma_list, [TRACKED],
        "only allow the listed language features to be enabled in code (comma separated)"),
    always_encode_mir: bool = (false, parse_bool, [TRACKED],
        "encode MIR of all functions into the crate metadata (default: no)"),
    annotate_moves: AnnotateMoves = (AnnotateMoves::Disabled, parse_annotate_moves, [TRACKED],
        "emit debug info for compiler-generated move and copy operations \
        to make them visible in profilers. Can be a boolean or a size limit in bytes (default: disabled)"),
    assert_incr_state: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "assert that the incremental cache is in given state: \
         either `loaded` or `not-loaded`."),
    assume_incomplete_release: bool = (false, parse_bool, [TRACKED],
        "make cfg(version) treat the current version as incomplete (default: no)"),
    autodiff: Vec<crate::config::AutoDiff> = (Vec::new(), parse_autodiff, [TRACKED],
        "a list of autodiff flags to enable
        Mandatory setting:
        `=Enable`
        Optional extra settings:
        `=PrintTA`
        `=PrintAA`
        `=PrintPerf`
        `=PrintSteps`
        `=PrintModBefore`
        `=PrintModAfter`
        `=PrintModFinal`
        `=PrintPasses`,
        `=NoPostopt`
        `=LooseTypes`
        `=Inline`
        Multiple options can be combined with commas."),
    #[rustc_lint_opt_deny_field_access("use `Session::binary_dep_depinfo` instead of this field")]
    binary_dep_depinfo: bool = (false, parse_bool, [TRACKED],
        "include artifacts (sysroot, crate dependencies) used during compilation in dep-info \
        (default: no)"),
    box_noalias: bool = (true, parse_bool, [TRACKED],
        "emit noalias metadata for box (default: yes)"),
    branch_protection: Option<BranchProtection> = (None, parse_branch_protection, [TRACKED TARGET_MODIFIER],
        "set options for branch target identification and pointer authentication on AArch64"),
    build_sdylib_interface: bool = (false, parse_bool, [UNTRACKED],
        "whether the stable interface is being built"),
    cache_proc_macros: bool = (false, parse_bool, [TRACKED],
        "cache the results of derive proc macro invocations (potentially unsound!) (default: no"),
    cf_protection: CFProtection = (CFProtection::None, parse_cfprotection, [TRACKED],
        "instrument control-flow architecture protection"),
    check_cfg_all_expected: bool = (false, parse_bool, [UNTRACKED],
        "show all expected values in check-cfg diagnostics (default: no)"),
    checksum_hash_algorithm: Option<SourceFileHashAlgorithm> = (None, parse_cargo_src_file_hash, [TRACKED],
        "hash algorithm of source files used to check freshness in cargo (`blake3` or `sha256`)"),
    codegen_backend: Option<String> = (None, parse_opt_string, [TRACKED],
        "the backend to use"),
    codegen_source_order: bool = (false, parse_bool, [UNTRACKED],
        "emit mono items in the order of spans in source files (default: no)"),
    contract_checks: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "emit runtime checks for contract pre- and post-conditions (default: no)"),
    coverage_options: CoverageOptions = (CoverageOptions::default(), parse_coverage_options, [TRACKED],
        "control details of coverage instrumentation"),
    crate_attr: Vec<String> = (Vec::new(), parse_string_push, [TRACKED],
        "inject the given attribute in the crate"),
    cross_crate_inline_threshold: InliningThreshold = (InliningThreshold::Sometimes(100), parse_inlining_threshold, [TRACKED],
        "threshold to allow cross crate inlining of functions"),
    debug_info_for_profiling: bool = (false, parse_bool, [TRACKED],
        "emit discriminators and other data necessary for AutoFDO"),
    debug_info_type_line_numbers: bool = (false, parse_bool, [TRACKED],
        "emit type and line information for additional data types (default: no)"),
    debuginfo_compression: DebugInfoCompression = (DebugInfoCompression::None, parse_debuginfo_compression, [TRACKED],
        "compress debug info sections (none, zlib, zstd, default: none)"),
    deduplicate_diagnostics: bool = (true, parse_bool, [UNTRACKED],
        "deduplicate identical diagnostics (default: yes)"),
    default_visibility: Option<SymbolVisibility> = (None, parse_opt_symbol_visibility, [TRACKED],
        "overrides the `default_visibility` setting of the target"),
    dep_info_omit_d_target: bool = (false, parse_bool, [TRACKED],
        "in dep-info output, omit targets for tracking dependencies of the dep-info files \
        themselves (default: no)"),
    direct_access_external_data: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "Direct or use GOT indirect to reference external data symbols"),
    dual_proc_macros: bool = (false, parse_bool, [TRACKED],
        "load proc macros for both target and host, but only link to the target (default: no)"),
    dump_dep_graph: bool = (false, parse_bool, [UNTRACKED],
        "dump the dependency graph to $RUST_DEP_GRAPH (default: /tmp/dep_graph.gv) \
        (default: no)"),
    dump_mir: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "dump MIR state to file.
        `val` is used to select which passes and functions to dump. For example:
        `all` matches all passes and functions,
        `foo` matches all passes for functions whose name contains 'foo',
        `foo & ConstProp` only the 'ConstProp' pass for function names containing 'foo',
        `foo | bar` all passes for function names containing 'foo' or 'bar'."),
    dump_mir_dataflow: bool = (false, parse_bool, [UNTRACKED],
        "in addition to `.mir` files, create graphviz `.dot` files with dataflow results \
        (default: no)"),
    dump_mir_dir: String = ("mir_dump".to_string(), parse_string, [UNTRACKED],
        "the directory the MIR is dumped into (default: `mir_dump`)"),
    dump_mir_exclude_alloc_bytes: bool = (false, parse_bool, [UNTRACKED],
        "exclude the raw bytes of allocations when dumping MIR (used in tests) (default: no)"),
    dump_mir_exclude_pass_number: bool = (false, parse_bool, [UNTRACKED],
        "exclude the pass number when dumping MIR (used in tests) (default: no)"),
    dump_mir_graphviz: bool = (false, parse_bool, [UNTRACKED],
        "in addition to `.mir` files, create graphviz `.dot` files (default: no)"),
    dump_mono_stats: SwitchWithOptPath = (SwitchWithOptPath::Disabled,
        parse_switch_with_opt_path, [UNTRACKED],
        "output statistics about monomorphization collection"),
    dump_mono_stats_format: DumpMonoStatsFormat = (DumpMonoStatsFormat::Markdown, parse_dump_mono_stats, [UNTRACKED],
        "the format to use for -Z dump-mono-stats (`markdown` (default) or `json`)"),
    #[rustc_lint_opt_deny_field_access("use `Session::dwarf_version` instead of this field")]
    dwarf_version: Option<u32> = (None, parse_opt_number, [TRACKED],
        "version of DWARF debug information to emit (default: 2 or 4, depending on platform)"),
    dylib_lto: bool = (false, parse_bool, [UNTRACKED],
        "enables LTO for dylib crate type"),
    eagerly_emit_delayed_bugs: bool = (false, parse_bool, [UNTRACKED],
        "emit delayed bugs eagerly as errors instead of stashing them and emitting \
        them only if an error has not been emitted"),
    ehcont_guard: bool = (false, parse_bool, [TRACKED],
        "generate Windows EHCont Guard tables"),
    embed_metadata: bool = (true, parse_bool, [TRACKED],
        "embed metadata in rlibs and dylibs (default: yes)"),
    embed_source: bool = (false, parse_bool, [TRACKED],
        "embed source text in DWARF debug sections (default: no)"),
    emit_stack_sizes: bool = (false, parse_bool, [UNTRACKED],
        "emit a section containing stack size metadata (default: no)"),
    emscripten_wasm_eh: bool = (true, parse_bool, [TRACKED],
        "Use WebAssembly error handling for wasm32-unknown-emscripten"),
    enforce_type_length_limit: bool = (false, parse_bool, [TRACKED],
        "enforce the type length limit when monomorphizing instances in codegen"),
    experimental_default_bounds: bool = (false, parse_bool, [TRACKED],
        "enable default bounds for experimental group of auto traits"),
    export_executable_symbols: bool = (false, parse_bool, [TRACKED],
        "export symbols from executables, as if they were dynamic libraries"),
    external_clangrt: bool = (false, parse_bool, [UNTRACKED],
        "rely on user specified linker commands to find clangrt"),
    extra_const_ub_checks: bool = (false, parse_bool, [TRACKED],
        "turns on more checks to detect const UB, which can be slow (default: no)"),
    #[rustc_lint_opt_deny_field_access("use `Session::fewer_names` instead of this field")]
    fewer_names: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "reduce memory use by retaining fewer names within compilation artifacts (LLVM-IR) \
        (default: no)"),
    fixed_x18: bool = (false, parse_bool, [TRACKED TARGET_MODIFIER],
        "make the x18 register reserved on AArch64 (default: no)"),
    flatten_format_args: bool = (true, parse_bool, [TRACKED],
        "flatten nested format_args!() and literals into a simplified format_args!() call \
        (default: yes)"),
    fmt_debug: FmtDebug = (FmtDebug::Full, parse_fmt_debug, [TRACKED],
        "how detailed `#[derive(Debug)]` should be. `full` prints types recursively, \
        `shallow` prints only type names, `none` prints nothing and disables `{:?}`. (default: `full`)"),
    force_unstable_if_unmarked: bool = (false, parse_bool, [TRACKED],
        "force all crates to be `rustc_private` unstable (default: no)"),
    function_return: FunctionReturn = (FunctionReturn::default(), parse_function_return, [TRACKED],
        "replace returns with jumps to `__x86_return_thunk` (default: `keep`)"),
    function_sections: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "whether each function should go in its own section"),
    future_incompat_test: bool = (false, parse_bool, [UNTRACKED],
        "forces all lints to be future incompatible, used for internal testing (default: no)"),
    graphviz_dark_mode: bool = (false, parse_bool, [UNTRACKED],
        "use dark-themed colors in graphviz output (default: no)"),
    graphviz_font: String = ("Courier, monospace".to_string(), parse_string, [UNTRACKED],
        "use the given `fontname` in graphviz output; can be overridden by setting \
        environment variable `RUSTC_GRAPHVIZ_FONT` (default: `Courier, monospace`)"),
    has_thread_local: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "explicitly enable the `cfg(target_thread_local)` directive"),
    help: bool = (false, parse_no_value, [UNTRACKED], "Print unstable compiler options"),
    higher_ranked_assumptions: bool = (false, parse_bool, [TRACKED],
        "allow deducing higher-ranked outlives assumptions from coroutines when proving auto traits"),
    hint_mostly_unused: bool = (false, parse_bool, [TRACKED],
        "hint that most of this crate will go unused, to minimize work for uncalled functions"),
    human_readable_cgu_names: bool = (false, parse_bool, [TRACKED],
        "generate human-readable, predictable names for codegen units (default: no)"),
    identify_regions: bool = (false, parse_bool, [UNTRACKED],
        "display unnamed regions as `'<id>`, using a non-ident unique id (default: no)"),
    ignore_directory_in_diagnostics_source_blocks: Vec<String> = (Vec::new(), parse_string_push, [UNTRACKED],
        "do not display the source code block in diagnostics for files in the directory"),
    incremental_ignore_spans: bool = (false, parse_bool, [TRACKED],
        "ignore spans during ICH computation -- used for testing (default: no)"),
    incremental_info: bool = (false, parse_bool, [UNTRACKED],
        "print high-level information about incremental reuse (or the lack thereof) \
        (default: no)"),
    incremental_verify_ich: bool = (false, parse_bool, [UNTRACKED],
        "verify extended properties for incr. comp. (default: no):
        - hashes of green query instances
        - hash collisions of query keys
        - hash collisions when creating dep-nodes"),
    indirect_branch_cs_prefix: bool = (false, parse_bool, [TRACKED TARGET_MODIFIER],
        "add `cs` prefix to `call` and `jmp` to indirect thunks (default: no)"),
    inline_llvm: bool = (true, parse_bool, [TRACKED],
        "enable LLVM inlining (default: yes)"),
    inline_mir: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "enable MIR inlining (default: no)"),
    inline_mir_forwarder_threshold: Option<usize> = (None, parse_opt_number, [TRACKED],
        "inlining threshold when the caller is a simple forwarding function (default: 30)"),
    inline_mir_hint_threshold: Option<usize> = (None, parse_opt_number, [TRACKED],
        "inlining threshold for functions with inline hint (default: 100)"),
    inline_mir_preserve_debug: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "when MIR inlining, whether to preserve debug info for callee variables \
        (default: preserve for debuginfo != None, otherwise remove)"),
    inline_mir_threshold: Option<usize> = (None, parse_opt_number, [TRACKED],
        "a default MIR inlining threshold (default: 50)"),
    input_stats: bool = (false, parse_bool, [UNTRACKED],
        "print some statistics about AST and HIR (default: no)"),
    instrument_mcount: bool = (false, parse_bool, [TRACKED],
        "insert function instrument code for mcount-based tracing (default: no)"),
    instrument_xray: Option<InstrumentXRay> = (None, parse_instrument_xray, [TRACKED],
        "insert function instrument code for XRay-based tracing (default: no)
         Optional extra settings:
         `=always`
         `=never`
         `=ignore-loops`
         `=instruction-threshold=N`
         `=skip-entry`
         `=skip-exit`
         Multiple options can be combined with commas."),
    large_data_threshold: Option<u64> = (None, parse_opt_number, [TRACKED],
        "set the threshold for objects to be stored in a \"large data\" section \
         (only effective with -Ccode-model=medium, default: 65536)"),
    layout_seed: Option<u64> = (None, parse_opt_number, [TRACKED],
        "seed layout randomization"),
    link_directives: bool = (true, parse_bool, [TRACKED],
        "honor #[link] directives in the compiled crate (default: yes)"),
    link_native_libraries: bool = (true, parse_bool, [UNTRACKED],
        "link native libraries in the linker invocation (default: yes)"),
    link_only: bool = (false, parse_bool, [TRACKED],
        "link the `.rlink` file generated by `-Z no-link` (default: no)"),
    lint_llvm_ir: bool = (false, parse_bool, [TRACKED],
        "lint LLVM IR (default: no)"),
    lint_mir: bool = (false, parse_bool, [UNTRACKED],
        "lint MIR before and after each transformation"),
    llvm_module_flag: Vec<(String, u32, String)> = (Vec::new(), parse_llvm_module_flag, [TRACKED],
        "a list of module flags to pass to LLVM (space separated)"),
    llvm_plugins: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "a list LLVM plugins to enable (space separated)"),
    llvm_time_trace: bool = (false, parse_bool, [UNTRACKED],
        "generate JSON tracing data file from LLVM data (default: no)"),
    location_detail: LocationDetail = (LocationDetail::all(), parse_location_detail, [TRACKED],
        "what location details should be tracked when using caller_location, either \
        `none`, or a comma separated list of location details, for which \
        valid options are `file`, `line`, and `column` (default: `file,line,column`)"),
    ls: Vec<String> = (Vec::new(), parse_list, [UNTRACKED],
        "decode and print various parts of the crate metadata for a library crate \
        (space separated)"),
    macro_backtrace: bool = (false, parse_bool, [UNTRACKED],
        "show macro backtraces (default: no)"),
    macro_stats: bool = (false, parse_bool, [UNTRACKED],
        "print some statistics about macro expansions (default: no)"),
    maximal_hir_to_mir_coverage: bool = (false, parse_bool, [TRACKED],
        "save as much information as possible about the correspondence between MIR and HIR \
        as source scopes (default: no)"),
    merge_functions: Option<MergeFunctions> = (None, parse_merge_functions, [TRACKED],
        "control the operation of the MergeFunctions LLVM pass, taking \
        the same values as the target option of the same name"),
    meta_stats: bool = (false, parse_bool, [UNTRACKED],
        "gather metadata statistics (default: no)"),
    metrics_dir: Option<PathBuf> = (None, parse_opt_pathbuf, [UNTRACKED],
        "the directory metrics emitted by rustc are dumped into (implicitly enables default set of metrics)"),
    min_function_alignment: Option<Align> = (None, parse_align, [TRACKED],
        "align all functions to at least this many bytes. Must be a power of 2"),
    min_recursion_limit: Option<usize> = (None, parse_opt_number, [TRACKED],
        "set a minimum recursion limit (final limit = max(this, recursion_limit_from_crate))"),
    mir_emit_retag: bool = (false, parse_bool, [TRACKED],
        "emit Retagging MIR statements, interpreted e.g., by miri; implies -Zmir-opt-level=0 \
        (default: no)"),
    mir_enable_passes: Vec<(String, bool)> = (Vec::new(), parse_list_with_polarity, [TRACKED],
        "use like `-Zmir-enable-passes=+DestinationPropagation,-InstSimplify`. Forces the \
        specified passes to be enabled, overriding all other checks. In particular, this will \
        enable unsound (known-buggy and hence usually disabled) passes without further warning! \
        Passes that are not specified are enabled or disabled by other flags as usual."),
    mir_include_spans: MirIncludeSpans = (MirIncludeSpans::default(), parse_mir_include_spans, [UNTRACKED],
        "include extra comments in mir pretty printing, like line numbers and statement indices, \
         details about types, etc. (boolean for all passes, 'nll' to enable in NLL MIR only, default: 'nll')"),
    mir_opt_bisect_limit: Option<usize> = (None, parse_opt_number, [TRACKED],
        "limit the number of MIR optimization pass executions (global across all bodies). \
        Pass executions after this limit are skipped and reported. (default: no limit)"),
    #[rustc_lint_opt_deny_field_access("use `Session::mir_opt_level` instead of this field")]
    mir_opt_level: Option<usize> = (None, parse_opt_number, [TRACKED],
        "MIR optimization level (0-4; default: 1 in non optimized builds and 2 in optimized builds)"),
    mir_preserve_ub: bool = (false, parse_bool, [TRACKED],
        "keep place mention statements and reads in trivial SwitchInt terminators, which are interpreted \
        e.g., by miri; implies -Zmir-opt-level=0 (default: no)"),
    mir_strip_debuginfo: MirStripDebugInfo = (MirStripDebugInfo::None, parse_mir_strip_debuginfo, [TRACKED],
        "Whether to remove some of the MIR debug info from methods.  Default: None"),
    move_size_limit: Option<usize> = (None, parse_opt_number, [TRACKED],
        "the size at which the `large_assignments` lint starts to be emitted"),
    mutable_noalias: bool = (true, parse_bool, [TRACKED],
        "emit noalias metadata for mutable references (default: yes)"),
    namespaced_crates: bool = (false, parse_bool, [TRACKED],
        "allow crates to be namespaced by other crates (default: no)"),
    next_solver: NextSolverConfig = (NextSolverConfig::default(), parse_next_solver_config, [TRACKED],
        "enable and configure the next generation trait solver used by rustc"),
    nll_facts: bool = (false, parse_bool, [UNTRACKED],
        "dump facts from NLL analysis into side files (default: no)"),
    nll_facts_dir: String = ("nll-facts".to_string(), parse_string, [UNTRACKED],
        "the directory the NLL facts are dumped into (default: `nll-facts`)"),
    no_analysis: bool = (false, parse_no_value, [UNTRACKED],
        "parse and expand the source, but run no analysis"),
    no_codegen: bool = (false, parse_no_value, [TRACKED_NO_CRATE_HASH],
        "run all passes except codegen; no output"),
    no_generate_arange_section: bool = (false, parse_no_value, [TRACKED],
        "omit DWARF address ranges that give faster lookups"),
    no_implied_bounds_compat: bool = (false, parse_bool, [TRACKED],
        "disable the compatibility version of the `implied_bounds_ty` query"),
    no_leak_check: bool = (false, parse_no_value, [UNTRACKED],
        "disable the 'leak check' for subtyping; unsound, but useful for tests"),
    no_link: bool = (false, parse_no_value, [TRACKED],
        "compile without linking"),
    no_parallel_backend: bool = (false, parse_no_value, [UNTRACKED],
        "run LLVM in non-parallel mode (while keeping codegen-units and ThinLTO)"),
    no_profiler_runtime: bool = (false, parse_no_value, [TRACKED],
        "prevent automatic injection of the profiler_builtins crate"),
    no_steal_thir: bool = (false, parse_bool, [UNTRACKED],
        "don't steal the THIR when we're done with it; useful for rustc drivers (default: no)"),
    no_trait_vptr: bool = (false, parse_no_value, [TRACKED],
        "disable generation of trait vptr in vtable for upcasting"),
    no_unique_section_names: bool = (false, parse_bool, [TRACKED],
        "do not use unique names for text and data sections when -Z function-sections is used"),
    normalize_docs: bool = (false, parse_bool, [TRACKED],
        "normalize associated items in rustdoc when generating documentation"),
    offload: Vec<crate::config::Offload> = (Vec::new(), parse_offload, [TRACKED],
        "a list of offload flags to enable
        Mandatory setting:
        `=Enable`
        Currently the only option available"),
    on_broken_pipe: OnBrokenPipe = (OnBrokenPipe::Default, parse_on_broken_pipe, [TRACKED],
        "behavior of std::io::ErrorKind::BrokenPipe (SIGPIPE)"),
    osx_rpath_install_name: bool = (false, parse_bool, [TRACKED],
        "pass `-install_name @rpath/...` to the macOS linker (default: no)"),
    packed_bundled_libs: bool = (false, parse_bool, [TRACKED],
        "change rlib format to store native libraries as archives"),
    panic_abort_tests: bool = (false, parse_bool, [TRACKED],
        "support compiling tests with panic=abort (default: no)"),
    panic_in_drop: PanicStrategy = (PanicStrategy::Unwind, parse_panic_strategy, [TRACKED],
        "panic strategy for panics in drops"),
    parse_crate_root_only: bool = (false, parse_bool, [UNTRACKED],
        "parse the crate root file only; do not parse other files, compile, assemble, or link \
        (default: no)"),
    patchable_function_entry: PatchableFunctionEntry = (PatchableFunctionEntry::default(), parse_patchable_function_entry, [TRACKED],
        "nop padding at function entry"),
    plt: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "whether to use the PLT when calling into shared libraries;
        only has effect for PIC code on systems with ELF binaries
        (default: PLT is disabled if full relro is enabled on x86_64)"),
    polonius: Polonius = (Polonius::default(), parse_polonius, [TRACKED],
        "enable polonius-based borrow-checker (default: no)"),
    pre_link_arg: (/* redirected to pre_link_args */) = ((), parse_string_push, [UNTRACKED],
        "a single extra argument to prepend the linker invocation (can be used several times)"),
    pre_link_args: Vec<String> = (Vec::new(), parse_list, [UNTRACKED],
        "extra arguments to prepend to the linker invocation (space separated)"),
    precise_enum_drop_elaboration: bool = (true, parse_bool, [TRACKED],
        "use a more precise version of drop elaboration for matches on enums (default: yes). \
        This results in better codegen, but has caused miscompilations on some tier 2 platforms. \
        See #77382 and #74551."),
    #[rustc_lint_opt_deny_field_access("use `Session::print_codegen_stats` instead of this field")]
    print_codegen_stats: bool = (false, parse_bool, [UNTRACKED],
        "print codegen statistics (default: no)"),
    print_llvm_passes: bool = (false, parse_bool, [UNTRACKED],
        "print the LLVM optimization passes being run (default: no)"),
    print_mono_items: bool = (false, parse_bool, [UNTRACKED],
        "print the result of the monomorphization collection pass (default: no)"),
    print_type_sizes: bool = (false, parse_bool, [UNTRACKED],
        "print layout information for each type encountered (default: no)"),
    proc_macro_backtrace: bool = (false, parse_bool, [UNTRACKED],
         "show backtraces for panics during proc-macro execution (default: no)"),
    proc_macro_execution_strategy: ProcMacroExecutionStrategy = (ProcMacroExecutionStrategy::SameThread,
        parse_proc_macro_execution_strategy, [UNTRACKED],
        "how to run proc-macro code (default: same-thread)"),
    profile_closures: bool = (false, parse_no_value, [UNTRACKED],
        "profile size of closures"),
    profile_sample_use: Option<PathBuf> = (None, parse_opt_pathbuf, [TRACKED],
        "use the given `.prof` file for sampled profile-guided optimization (also known as AutoFDO)"),
    profiler_runtime: String = (String::from("profiler_builtins"), parse_string, [TRACKED],
        "name of the profiler runtime crate to automatically inject (default: `profiler_builtins`)"),
    query_dep_graph: bool = (false, parse_bool, [UNTRACKED],
        "enable queries of the dependency graph for regression testing (default: no)"),
    randomize_layout: bool = (false, parse_bool, [TRACKED],
        "randomize the layout of types (default: no)"),
    reg_struct_return: bool = (false, parse_bool, [TRACKED TARGET_MODIFIER],
        "On x86-32 targets, it overrides the default ABI to return small structs in registers.
        It is UNSOUND to link together crates that use different values for this flag!"),
    regparm: Option<u32> = (None, parse_opt_number, [TRACKED TARGET_MODIFIER],
        "On x86-32 targets, setting this to N causes the compiler to pass N arguments \
        in registers EAX, EDX, and ECX instead of on the stack for\
        \"C\", \"cdecl\", and \"stdcall\" fn.\
        It is UNSOUND to link together crates that use different values for this flag!"),
    relax_elf_relocations: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "whether ELF relocations can be relaxed"),
    remap_cwd_prefix: Option<PathBuf> = (None, parse_opt_pathbuf, [TRACKED],
        "remap paths under the current working directory to this path prefix"),
    remark_dir: Option<PathBuf> = (None, parse_opt_pathbuf, [UNTRACKED],
        "directory into which to write optimization remarks (if not specified, they will be \
written to standard error output)"),
    retpoline: bool = (false, parse_bool, [TRACKED TARGET_MODIFIER],
        "enables retpoline-indirect-branches and retpoline-indirect-calls target features (default: no)"),
    retpoline_external_thunk: bool = (false, parse_bool, [TRACKED TARGET_MODIFIER],
        "enables retpoline-external-thunk, retpoline-indirect-branches and retpoline-indirect-calls \
        target features (default: no)"),
    #[rustc_lint_opt_deny_field_access("use `Session::sanitizers()` instead of this field")]
    sanitizer: SanitizerSet = (SanitizerSet::empty(), parse_sanitizers, [TRACKED TARGET_MODIFIER],
        "use a sanitizer"),
    sanitizer_cfi_canonical_jump_tables: Option<bool> = (Some(true), parse_opt_bool, [TRACKED],
        "enable canonical jump tables (default: yes)"),
    sanitizer_cfi_generalize_pointers: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "enable generalizing pointer types (default: no)"),
    sanitizer_cfi_normalize_integers: Option<bool> = (None, parse_opt_bool, [TRACKED TARGET_MODIFIER],
        "enable normalizing integer types (default: no)"),
    sanitizer_dataflow_abilist: Vec<String> = (Vec::new(), parse_comma_list, [TRACKED],
        "additional ABI list files that control how shadow parameters are passed (comma separated)"),
    sanitizer_kcfi_arity: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "enable KCFI arity indicator (default: no)"),
    sanitizer_memory_track_origins: usize = (0, parse_sanitizer_memory_track_origins, [TRACKED],
        "enable origins tracking in MemorySanitizer"),
    sanitizer_recover: SanitizerSet = (SanitizerSet::empty(), parse_sanitizers, [TRACKED],
        "enable recovery for selected sanitizers"),
    saturating_float_casts: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "make float->int casts UB-free: numbers outside the integer type's range are clipped to \
        the max/min integer respectively, and NaN is mapped to 0 (default: yes)"),
    self_profile: SwitchWithOptPath = (SwitchWithOptPath::Disabled,
        parse_switch_with_opt_path, [UNTRACKED],
        "run the self profiler and output the raw event data"),
    self_profile_counter: String = ("wall-time".to_string(), parse_string, [UNTRACKED],
        "counter used by the self profiler (default: `wall-time`), one of:
        `wall-time` (monotonic clock, i.e. `std::time::Instant`)
        `instructions:u` (retired instructions, userspace-only)
        `instructions-minus-irqs:u` (subtracting hardware interrupt counts for extra accuracy)"
    ),
    /// keep this in sync with the event filter names in librustc_data_structures/profiling.rs
    self_profile_events: Option<Vec<String>> = (None, parse_opt_comma_list, [UNTRACKED],
        "specify the events recorded by the self profiler;
        for example: `-Z self-profile-events=default,query-keys`
        all options: none, all, default, generic-activity, query-provider, query-cache-hit
                     query-blocked, incr-cache-load, incr-result-hashing, query-keys, function-args, args, llvm, artifact-sizes"),
    share_generics: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "make the current crate share its generic instantiations"),
    shell_argfiles: bool = (false, parse_bool, [UNTRACKED],
        "allow argument files to be specified with POSIX \"shell-style\" argument quoting"),
    simulate_remapped_rust_src_base: Option<PathBuf> = (None, parse_opt_pathbuf, [TRACKED],
        "simulate the effect of remap-debuginfo = true at bootstrapping by remapping path \
        to rust's source base directory. only meant for testing purposes"),
    small_data_threshold: Option<usize> = (None, parse_opt_number, [TRACKED],
        "Set the threshold for objects to be stored in a \"small data\" section"),
    span_debug: bool = (false, parse_bool, [UNTRACKED],
        "forward proc_macro::Span's `Debug` impl to `Span`"),
    /// o/w tests have closure@path
    span_free_formats: bool = (false, parse_bool, [UNTRACKED],
        "exclude spans when debug-printing compiler state (default: no)"),
    split_dwarf_inlining: bool = (false, parse_bool, [TRACKED],
        "provide minimal debug info in the object/executable to facilitate online \
         symbolication/stack traces in the absence of .dwo/.dwp files when using Split DWARF"),
    split_dwarf_kind: SplitDwarfKind = (SplitDwarfKind::Split, parse_split_dwarf_kind, [TRACKED],
        "split dwarf variant (only if -Csplit-debuginfo is enabled and on relevant platform)
        (default: `split`)

        `split`: sections which do not require relocation are written into a DWARF object (`.dwo`)
                 file which is ignored by the linker
        `single`: sections which do not require relocation are written into object file but ignored
                  by the linker"),
    split_dwarf_out_dir : Option<PathBuf> = (None, parse_opt_pathbuf, [TRACKED],
        "location for writing split DWARF objects (`.dwo`) if enabled"),
    split_lto_unit: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "enable LTO unit splitting (default: no)"),
    src_hash_algorithm: Option<SourceFileHashAlgorithm> = (None, parse_src_file_hash, [TRACKED],
        "hash algorithm of source files in debug info (`md5`, `sha1`, or `sha256`)"),
    #[rustc_lint_opt_deny_field_access("use `Session::stack_protector` instead of this field")]
    stack_protector: StackProtector = (StackProtector::None, parse_stack_protector, [TRACKED],
        "control stack smash protection strategy (`rustc --print stack-protector-strategies` for details)"),
    staticlib_allow_rdylib_deps: bool = (false, parse_bool, [TRACKED],
        "allow staticlibs to have rust dylib dependencies"),
    staticlib_prefer_dynamic: bool = (false, parse_bool, [TRACKED],
        "prefer dynamic linking to static linking for staticlibs (default: no)"),
    strict_init_checks: bool = (false, parse_bool, [TRACKED],
        "control if mem::uninitialized and mem::zeroed panic on more UB"),
    #[rustc_lint_opt_deny_field_access("use `Session::teach` instead of this field")]
    teach: bool = (false, parse_bool, [TRACKED],
        "show extended diagnostic help (default: no)"),
    temps_dir: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "the directory the intermediate files are written to"),
    terminal_urls: TerminalUrl = (TerminalUrl::No, parse_terminal_url, [UNTRACKED],
        "use the OSC 8 hyperlink terminal specification to print hyperlinks in the compiler output"),
    #[rustc_lint_opt_deny_field_access("use `Session::lto` instead of this field")]
    thinlto: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "enable ThinLTO when possible"),
    /// We default to 1 here since we want to behave like
    /// a sequential compiler for now. This'll likely be adjusted
    /// in the future. Note that -Zthreads=0 is the way to get
    /// the num_cpus behavior.
    #[rustc_lint_opt_deny_field_access("use `Session::threads` instead of this field")]
    threads: usize = (1, parse_threads, [UNTRACKED],
        "use a thread pool with N threads"),
    time_llvm_passes: bool = (false, parse_bool, [UNTRACKED],
        "measure time of each LLVM pass (default: no)"),
    time_passes: bool = (false, parse_bool, [UNTRACKED],
        "measure time of each rustc pass (default: no)"),
    time_passes_format: TimePassesFormat = (TimePassesFormat::Text, parse_time_passes_format, [UNTRACKED],
        "the format to use for -Z time-passes (`text` (default) or `json`)"),
    tiny_const_eval_limit: bool = (false, parse_bool, [TRACKED],
        "sets a tiny, non-configurable limit for const eval; useful for compiler tests"),
    #[rustc_lint_opt_deny_field_access("use `Session::tls_model` instead of this field")]
    tls_model: Option<TlsModel> = (None, parse_tls_model, [TRACKED],
        "choose the TLS model to use (`rustc --print tls-models` for details)"),
    trace_macros: bool = (false, parse_bool, [UNTRACKED],
        "for every macro invocation, print its name and arguments (default: no)"),
    track_diagnostics: bool = (false, parse_bool, [UNTRACKED],
        "tracks where in rustc a diagnostic was emitted"),
    translate_remapped_path_to_local_path: bool = (true, parse_bool, [TRACKED],
        "translate remapped paths into local paths when possible (default: yes)"),
    trap_unreachable: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "generate trap instructions for unreachable intrinsics (default: use target setting, usually yes)"),
    treat_err_as_bug: Option<NonZero<usize>> = (None, parse_treat_err_as_bug, [TRACKED],
        "treat the `val`th error that occurs as bug (default if not specified: 0 - don't treat errors as bugs. \
        default if specified without a value: 1 - treat the first error as bug)"),
    trim_diagnostic_paths: bool = (true, parse_bool, [UNTRACKED],
        "in diagnostics, use heuristics to shorten paths referring to items"),
    tune_cpu: Option<String> = (None, parse_opt_string, [TRACKED],
        "select processor to schedule for (`rustc --print target-cpus` for details)"),
    #[rustc_lint_opt_deny_field_access("use `TyCtxt::use_typing_mode_borrowck` instead of this field")]
    typing_mode_borrowck: bool = (false, parse_bool, [TRACKED],
        "enable `TypingMode::Borrowck`, changing the way opaque types are handled during MIR borrowck"),
    #[rustc_lint_opt_deny_field_access("use `Session::ub_checks` instead of this field")]
    ub_checks: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "emit runtime checks for Undefined Behavior (default: -Cdebug-assertions)"),
    ui_testing: bool = (false, parse_bool, [UNTRACKED],
        "emit compiler diagnostics in a form suitable for UI testing (default: no)"),
    uninit_const_chunk_threshold: usize = (16, parse_number, [TRACKED],
        "allow generating const initializers with mixed init/uninit chunks, \
        and set the maximum number of chunks for which this is allowed (default: 16)"),
    unleash_the_miri_inside_of_you: bool = (false, parse_bool, [TRACKED],
        "take the brakes off const evaluation. NOTE: this is unsound (default: no)"),
    unpretty: Option<String> = (None, parse_unpretty, [UNTRACKED],
        "present the input source, unstable (and less-pretty) variants;
        `normal`, `identified`,
        `expanded`, `expanded,identified`,
        `expanded,hygiene` (with internal representations),
        `ast-tree` (raw AST before expansion),
        `ast-tree,expanded` (raw AST after expansion),
        `hir` (the HIR), `hir,identified`,
        `hir,typed` (HIR with types for each node),
        `hir-tree` (dump the raw HIR),
        `thir-tree`, `thir-flat`,
        `mir` (the MIR), or `mir-cfg` (graphviz formatted MIR)"),
    unsound_mir_opts: bool = (false, parse_bool, [TRACKED],
        "enable unsound and buggy MIR optimizations (default: no)"),
    /// This name is kind of confusing: Most unstable options enable something themselves, while
    /// this just allows "normal" options to be feature-gated.
    ///
    /// The main check for `-Zunstable-options` takes place separately from the
    /// usual parsing of `-Z` options (see [`crate::config::nightly_options`]),
    /// so this boolean value is mostly used for enabling unstable _values_ of
    /// stable options. That separate check doesn't handle boolean values, so
    /// to avoid an inconsistent state we also forbid them here.
    #[rustc_lint_opt_deny_field_access("use `Session::unstable_options` instead of this field")]
    unstable_options: bool = (false, parse_no_value, [UNTRACKED],
        "adds unstable command line options to rustc interface (default: no)"),
    use_ctors_section: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "use legacy .ctors section for initializers rather than .init_array"),
    use_sync_unwind: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "Generate sync unwind tables instead of async unwind tables (default: no)"),
    validate_mir: bool = (false, parse_bool, [UNTRACKED],
        "validate MIR after each transformation"),
    verbose_asm: bool = (false, parse_bool, [TRACKED],
        "add descriptive comments from LLVM to the assembly (may change behavior) (default: no)"),
    #[rustc_lint_opt_deny_field_access("use `Session::verbose_internals` instead of this field")]
    verbose_internals: bool = (false, parse_bool, [TRACKED_NO_CRATE_HASH],
        "in general, enable more debug printouts (default: no)"),
    #[rustc_lint_opt_deny_field_access("use `Session::verify_llvm_ir` instead of this field")]
    verify_llvm_ir: bool = (false, parse_bool, [TRACKED],
        "verify LLVM IR (default: no)"),
    virtual_function_elimination: bool = (false, parse_bool, [TRACKED],
        "enables dead virtual function elimination optimization. \
        Requires `-Clto[=[fat,yes]]`"),
    wasi_exec_model: Option<WasiExecModel> = (None, parse_wasi_exec_model, [TRACKED],
        "whether to build a wasi command or reactor"),
    // This option only still exists to provide a more gradual transition path for people who need
    // the spec-complaint C ABI to be used.
    // FIXME remove this after a couple releases
    wasm_c_abi: () = ((), parse_wasm_c_abi, [TRACKED],
        "use spec-compliant C ABI for `wasm32-unknown-unknown` (deprecated, always enabled)"),
    write_long_types_to_disk: bool = (true, parse_bool, [UNTRACKED],
        "whether long type names should be written to files instead of being printed in errors"),
    // tidy-alphabetical-end

    // If you add a new option, please update:
    // - compiler/rustc_interface/src/tests.rs
    // - src/doc/unstable-book/src/compiler-flags
}
