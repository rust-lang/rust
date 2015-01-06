#compdef rustc

local -a _rustc_opts_switches _rustc_opts_lint _rustc_opts_debug

typeset -A opt_args

_rustc_debuginfo_levels=(
    "0[no debug info]"
    "1[line-tables only (for stacktraces and breakpoints)]"
    "2[full debug info with variable and type information (same as -g)]"
)

_rustc_crate_types=(
    'bin'
    'lib'
    'rlib'
    'dylib'
    'staticlib'
)

_rustc_emit_types=(
    'asm'
    'llvm-bc'
    'llvm-ir'
    'obj'
    'link'
    'dep-info'
)
_rustc_pretty_types=(
    'normal[un-annotated source]'
    'expanded[crates expanded]'
    'typed[crates  expanded,  with  type  annotations]'
    'identified[fully parenthesized, AST nodes and blocks with IDs]'
    'flowgraph[graphviz formatted flowgraph for node]:NODEID:'
)
_rustc_color_types=(
    'auto[colorize, if output goes to a tty (default)]'
    'always[always colorize output]'
    'never[never colorize output]'
)
_rustc_info_types=(
    'crate-name[Output the crate name and exit]'
    'file-names[Output the file(s) that would be written if compilation continued and exited]'
    'sysroot[Output the sysroot and exit]'
)

_rustc_opts_vals=(
    --crate-name='[Specify the name of the crate being built]'
    --crate-type='[Comma separated list of types of crates for the compiler to emit]:TYPES:_values -s "," "Crate types"  "$_rustc_crate_types[@]"'
    --emit='[Comma separated list of types of output for the compiler to emit]:TYPES:_values -s "," "Emit Targets" "$_rustc_emit_types[@]"'
    --cfg='[Configure the compilation environment]:SPEC:'
    --out-dir='[Write output to compiler-chosen filename in <dir>.  Ignored  if  -o  is  specified. (default the current directory)]:DIR:_files -/'
    -o'[Write output to <filename>. Ignored if more than one --emit is specified.]:FILENAME:_files'
    --pretty='[Pretty-print the input instead of compiling]::TYPE:_values "TYPES" "$_rustc_pretty_types[@]"'
    -L'[Add a directory to the library search path]:DIR:_files -/'
    --target='[Target triple cpu-manufacturer-kernel\[-os\] to compile]:TRIPLE:'
    --color='[Configure coloring of output]:CONF:_values "COLORS" "$_rustc_color_types[@]"'
    {-v,--version}'[Print version info and exit]::VERBOSE:(verbose)'
    --explain='[Provide a detailed explanation of an error message]:OPT:'
    --extern'[Specify where an external rust library is located]:ARG:'
    --print='[Comma separated list of compiler information to print on stdout]:TYPES:_values -s "," "Compiler Information" "$_rustc_info_types[@]"'
)

_rustc_opts_switches=(
    -g'[Equivalent to -C debuginfo=2]'
    {-h,--help}'[Display the help message]'
    {-V,--verbose}'[use verbose output]'
    -O'[Equivalent to -C opt-level=2]'
    --test'[Build a test harness]'
)


_rustc_opts_link=(
    'static[Path to the library to link statically]:PATH:_files -/'
    'dylib[Path to the library to link dynamically]:PATH:_files -/'
    'framework[Path to the library to link as a framework]:PATH:_files -/'
)

_rustc_opts_codegen=(
    'ar[Path to the archive utility to use when assembling archives.]:BIN:_path_files'
    'linker[Path to the linker utility to use when linking libraries, executables, and objects.]:BIN:_path_files'
    'link-args[A space-separated list of extra arguments to pass to the linker when the linker is invoked.]:ARGS:'
    'lto[Perform LLVM link-time optimizations]'
    'target-cpu[Selects a target processor. If the value is "help", then a list of  available  CPUs is printed.]:CPU:'
    'target-feature[A space-separated list of features to enable or disable for the target. A preceding "+" enables a feature while a preceding "-" disables it. Available features can be discovered through target-cpu=help.]:FEATURE:'
    'passes[A space-separated list of extra LLVM passes to run. A value of "list" will cause rustc to print all known passes and exit. The passes specified are appended at the end of the normal pass manager.]:LIST:'
    'llvm-args[A space-separated list of arguments to pass through to LLVM.]:ARGS:'
    'save-temps[If specified, the compiler will save more files (.bc, .o, .no-opt.bc) generated throughout compilation in the output directory.]'
    'rpath[If specified, then the rpath value for dynamic libraries will be set in either dynamic library or executable outputs.]'
    'no-prepopulate-passes[Suppresses pre-population of the LLVM pass manager that is run over the module.]'
    'no-vectorize-loops[Suppresses running the loop vectorization LLVM pass, regardless of optimization level.]'
    'no-vectorize-slp[Suppresses running the LLVM SLP vectorization pass, regardless of optimization level.]'
    'soft-float[Generates software floating point library calls instead of hardware instructions.]'
    'prefer-dynamic[Prefers dynamic linking to static linking.]'
    "no-integrated-as[Force usage of an external assembler rather than LLVM's integrated one.]"
    'no-redzone[disable the use of the redzone]'
    'relocation-model[The relocation model to use. (default: pic)]:MODEL:(pic static dynamic-no-pic)'
    'code-model[choose the code model to use (llc -code-model for details)]:MODEL:'
    'metadata[metadata to mangle symbol names with]:VAL:'
    'extra-filenames[extra data to put in each output filename]:VAL:'
    'codegen-units[divide crate into N units to optimize in parallel]:N:'
    'remark[print remarks for these optimization passes (space separated, or "all")]:TYPE:'
    'debuginfo[debug info emission level, 0 = no debug info, 1 = line tables only, 2 = full debug info with variable and type information]:LEVEL:_values "Debug Levels" "$_rustc_debuginfo_levels[@]"'
    'opt-level[Optimize with possible levels 0-3]:LEVEL:(0 1 2 3)'
    'help[Show all codegen options]'
)

_rustc_opts_lint=(
    'help[Show a list of all lints]'
    'box-pointers[(default: allow) use of owned (Box type) heap memory]'
    'experimental[(default: allow) detects use of #\[experimental\] items]'
    'fat-ptr-transmutes[(default: allow) detects transmutes of fat pointers]'
    'missing-docs[(default: allow) detects missing documentation for public members]'
    'unsafe-blocks[(default: allow) usage of an "unsafe" block]'
    'unstable[(default: allow) detects use of #\[unstable\] items (incl. items with no stability attribute)]'
    'unused-extern-crates[(default: allow) extern crates that are never used]'
    'unused-import-braces[(default: allow) unnecessary braces around an imported item]'
    'unused-qualifications[(default: allow) detects unnecessarily qualified names]'
    'unused-results[(default: allow) unused result of an expression in a statement]'
    'unused-typecasts[(default: allow) detects unnecessary type casts that can be removed]'
    'variant-size-differences[(default: allow) detects enums with widely varying variant sizes]'
    'dead-code[(default: warn) detect unused, unexported items]'
    'deprecated[(default: warn) detects use of #\[deprecated\] items]'
    'improper-ctypes[(default: warn) proper use of libc types in foreign modules]'
    'missing-copy-implementations[(default: warn) detects potentially-forgotten implementations of "Copy"]'
    'non-camel-case-types[(default: warn) types, variants, traits and type parameters should have camel case names]'
    'non-shorthand-field-patterns[(default: warn) using "Struct { x: x }" instead of "Struct { x }"]'
    'non-snake-case[(default: warn) methods, functions, lifetime parameters and modules should have snake case names]'
    'non-upper-case-globals[(default: warn) static constants should have uppercase identifiers]'
    'overflowing-literals[(default: warn) literal out of range for its type]'
    'path-statements[(default: warn) path statements with no effect]'
    'raw-pointer-deriving[(default: warn) uses of #\[derive\] with raw pointers are rarely correct]'
    'unknown-lints[(default: warn) unrecognized lint attribute]'
    'unreachable-code[(default: warn) detects unreachable code paths]'
    'unsigned-negation[(default: warn) using an unary minus operator on unsigned type]'
    'unused-allocation[(default: warn) detects unnecessary allocations that can be eliminated]'
    'unused-assignments[(default: warn) detect assignments that will never be read]'
    'unused-attributes[(default: warn) detects attributes that were not used by the compiler]'
    'unused-comparisons[(default: warn) comparisons made useless by limits of the types involved]'
    'unused-imports[(default: warn) imports that are never used]'
    'unused-must-use[(default: warn) unused result of a type flagged as must_use]'
    "unused-mut[(default: warn) detect mut variables which don't need to be mutable]"
    'unused-parens[(default: warn) "if", "match", "while" and "return" do not need parentheses]'
    'unused-unsafe[(default: warn) unnecessary use of an "unsafe" block]'
    'unused-variables[(default: warn) detect variables which are not used in any way]'
    'warnings[(default: warn) mass-change the level for lints which produce warnings]'
    'while-true[(default: warn) suggest using "loop { }" instead of "while true { }"]'
    "exceeding-bitshifts[(default: deny) shift exceeds the type's number of bits]"
    'unknown-crate-types[(default: deny) unknown crate type found in #\[crate_type\] directive]'
    'unknown-features[(default: deny) unknown features found in crate-level #\[feature\] directives]'
    'bad-style[non-camel-case-types, non-snake-case, non-upper-case-globals]'
    'unused[unused-imports, unused-variables, unused-assignments, dead-code, unused-mut, unreachable-code, unused-must-use, unused-unsafe, path-statements]'
)

_rustc_opts_debug=(
    'verbose[in general, enable more debug printouts]'
    'time-passes[measure time of each rustc pass]'
    'count-llvm-insns[count where LLVM instrs originate]'
    'time-llvm-passes[measure time of each LLVM pass]'
    'trans-stats[gather trans statistics]'
    'asm-comments[generate comments into the assembly (may change behavior)]'
    'no-verify[skip LLVM verification]'
    'borrowck-stats[gather borrowck statistics]'
    'no-landing-pads[omit landing pads for unwinding]'
    'debug-llvm[enable debug output from LLVM]'
    'show-span[show spans for compiler debugging]'
    'count-type-sizes[count the sizes of aggregate types]'
    'meta-stats[gather metadata statistics]'
    'print-link-args[Print the arguments passed to the linker]'
    'gc[Garbage collect shared data (experimental)]'
    'print-llvm-passes[Prints the llvm optimization passes being run]'
    'ast-json[Print the AST as JSON and halt]'
    'ast-json-noexpand[Print the pre-expansion AST as JSON and halt]'
    'ls[List the symbols defined by a library crate]'
    'save-analysis[Write syntax and type analysis information in addition to normal output]'
    'flowgraph-print-loans[Include loan analysis data in --pretty flowgraph output]'
    'flowgraph-print-moves[Include move analysis data in --pretty flowgraph output]'
    'flowgraph-print-assigns[Include assignment analysis data in --pretty flowgraph output]'
    'flowgraph-print-all[Include all dataflow analysis data in --pretty flowgraph output]'
    'print-regiion-graph[Prints region inference graph. Use with RUST_REGION_GRAPH=help for more info]'
    'parse-only[Parse only; do not compile, assemble, or link]'
    'no-trans[Run all passes except translation; no output]'
    'no-analysis[Parse and expand the source, but run no analysis]'
    'unstable-options[Adds unstable command line options to rustc interface]'
    'print-enum-sizes[Print the size of enums and their variants]'
)

_rustc_opts_fun_lint(){
    _values -s , 'options' \
        "$_rustc_opts_lint[@]"
}

_rustc_opts_fun_debug(){
    _values 'options' "$_rustc_opts_debug[@]"
}

_rustc_opts_fun_codegen(){
    _values 'options' "$_rustc_opts_codegen[@]"
}

_rustc_opts_fun_link(){
    _values 'options' "$_rustc_opts_link[@]"
}

_arguments -s :  \
    '(-W --warn)'{-W,--warn=}'[Set lint warnings]:lint options:_rustc_opts_fun_lint' \
    '(-A --allow)'{-A,--allow=}'[Set lint allowed]:lint options:_rustc_opts_fun_lint' \
    '(-D --deny)'{-D,--deny=}'[Set lint denied]:lint options:_rustc_opts_fun_lint' \
    '(-F --forbid)'{-F,--forbid=}'[Set lint forbidden]:lint options:_rustc_opts_fun_lint' \
    '*-Z[Set internal debugging options]:debug options:_rustc_opts_fun_debug' \
    '(-C --codegen)'{-C,--codegen}'[Set internal Codegen options]:codegen options:_rustc_opts_fun_codegen' \
    '*-l[Link the generated crates to the specified native library NAME. the optional KIND can be one of, static, dylib, or framework. If omitted, dylib is assumed.]:ARG:_rustc_opts_fun_link' \
    "$_rustc_opts_switches[@]" \
    "$_rustc_opts_vals[@]" \
    '::files:_files -g "*.rs"'
