/*                   */ #[inline(always)]
/*                   */ pub fn __incr_cov<T>(_region_loc: &str, /*index: u32,*/ result: T) -> T {
/*                   */     result
/*                   */ }
/*                   */
/*    -              */ fn main() {
/*    ┃              */     let mut countdown = 10;
/*    ┃              */     __incr_cov("block start",()); // Must increment before repeated while text expression
/*    :   I          */     while __incr_cov("while test", countdown > 0) { // span is just the while test expression
/*    :   ┃          */         countdown -= 1;
/*    :   ┃          */         // __incr_cov("while loop",()); // Counter not needed, but span is computed as "while test" minus "block start"
/*    :   ┃          */                                         // If while criteria is tested 11 times, and the outer block runs only once, 11-1 = 10
/*    :   ┃          */         // REMOVING COUNTER ASSUMES NO EARLY RETURN THOUGH.
/*    :   ┃          */         // I THINK WE CAN ONLY USE THE COUNTER EXPRESSION UP TO FIRST CONDITIONAL BLOCK, IF ANY (if, match, maybe any loop)
/*    ┃   -          */     }

    let mut countdown = 10;
    __incr_cov("after first while loop",());
    while __incr_cov("while test", countdown > 0) {
        countdown -= 1;
        // if __incr_cov("top of while loop", countdown < 5) {
        if countdown < 5 { // "top of while loop" = counter expression "while test" - "after first while loop"
            __incr_cov("top of if countdown < 5",());
            break;
        }
        countdown -= 2;
        // __incr_cov("after if countdown < 5 block", ());
        // "after if countdown < 5 block" = counter expression "top of while loop" - "top of if countdown < 5"
        // HOWEVER, WE CAN ONLY REMOVE THE COUNTER AND USE COUNTER EXPRESSION IF WE **KNOW** THAT THE BODY OF THE IF
        // WILL **ALWAYS** BREAK (OR RETURN, OR CONTINUE?)
        // AND THUS WE TREAT THE STATEMENTS FOLLOWING THE IF BLOCK AS IF THEY WERE AN ELSE BLOCK.
        // THAT'S A LOT TO ASK.

        // PERHAPS TREAT EARLY RETURNS AS A SPECIAL KIND OF COUNTER AND IF ANY ARE INVOKED BEFORE STATEMENTS AFTER THE BLOCK THAT CONTAINS THEM,
        // THEN SUBTRACT THOSE COUNTS FROM THE COUNT BEFORE THE BLOCK (AS WE DO HERE)? (SO ONE SET OF EXPRESSIONS MUST SUM ALL OF THE EARLY
        // RETURNS)
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
