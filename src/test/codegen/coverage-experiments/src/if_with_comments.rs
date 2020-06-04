/*                   */ #[inline(always)]
/*                   */ pub fn __incr_cov<T>(_region_loc: &str, /*index: u32,*/ result: T) -> T {
/*                   */     result
/*                   */ }
/*                   */
/*    -              */ fn main() {
/*    ┃              */     let mut countdown = 10;
/*    :   I          */     if __incr_cov("start", countdown > 0) { // span is from start of main()
/*    :   ┃          */         countdown -= 1;
/*    :   ┃          */         __incr_cov("if block",());
/*    ┃   -          */     }

    let mut countdown = 10;
    if __incr_cov("start", countdown > 0) {
        countdown -= 1;
        __incr_cov("if block",());
    } else if countdown > 5 { // counter expression "start" - "if block"
        countdown -= 2;
        __incr_cov("else if block",());
    } else {
        countdown -= 3;
        // __incr_cov("else block",()); // counter expression (countdown > 5 counter expression) - "else if block"
                                        // PLACED AT END OF ELSE BLOCK OR START OF FIRST CONDITIONAL BLOCK, IF ANY (PRESUMING POSSIBLE EARLY EXIT).
                                        // IF WE CAN GUARANTEE NO EARLY EXIT IN THIS BLOCK, THEN AT THE END IS FINE EVEN IF ELSE BLOCK CONTAINS OTHER CONDITIONS.
    }

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
