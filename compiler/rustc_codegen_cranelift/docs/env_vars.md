# List of env vars recognized by cg_clif

<dl>
    <dt>CG_CLIF_JIT_ARGS</dt>
    <dd>When JIT mode is enable pass these arguments to the program.</dd>
    <dt>CG_CLIF_INCR_CACHE_DISABLED</dt>
    <dd>Don't cache object files in the incremental cache. Useful during development of cg_clif
    to make it possible to use incremental mode for all analyses performed by rustc without caching
    object files when their content should have been changed by a change to cg_clif.</dd>
    <dt>CG_CLIF_DISPLAY_CG_TIME</dt>
    <dd>If "1", display the time it took to perform codegen for a crate</dd>
    <dt>CG_CLIF_FUNCTION_SECTIONS</dt>
    <dd>Use a single section for each function. This will often reduce the executable size at the
        cost of making linking significantly slower.</dd>
</dl>
