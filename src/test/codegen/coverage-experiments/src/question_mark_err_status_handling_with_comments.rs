/*                   */ #[inline(always)]
/*                   */ pub fn __incr_cov<T>(_region_loc: &str, /*index: u32,*/ result: T) -> T {
/*                   */     result
/*                   */ }
/*                   */
/*    -              */ fn main() {
/*    :   I          */     for countdown in __incr_cov("start", 10..0) { // span is just the while test expression
/*    :   ┃          */         let _ = countdown;
/*    :   ┃          */         __incr_cov("top of for", ());
/*    ┃   -          */     }
/*    -              */ }


// -Z unpretty=val -- present the input source, unstable (and less-pretty) variants;
// valid types are any of the types for `--pretty`, as well as:
// `expanded`, `expanded,identified`,
// `expanded,hygiene` (with internal representations),
// `everybody_loops` (all function bodies replaced with `loop {}`),
// `hir` (the HIR), `hir,identified`,
// `hir,typed` (HIR with types for each node),
// `hir-tree` (dump the raw HIR),
// `mir` (the MIR), or `mir-cfg` (graphviz formatted MIR)

// argument to `pretty` must be one of `normal`, `expanded`, `identified`, or `expanded,identified`
