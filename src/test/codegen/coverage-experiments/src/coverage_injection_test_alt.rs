/*                       */ use std::io::Error;
/*                       */ use std::io::ErrorKind;
/*                       */
/*                       */ /// Align Rust counter increment with with:
/*                       */ /// [‘llvm.instrprof.increment’ Intrinsic](https://llvm.org/docs/LangRef.html#llvm-instrprof-increment-intrinsic)
/*                       */ ///
/*                       */ /// declare void @llvm.instrprof.increment(i8* <name>, i64 <hash>, i32 <num-counters>, i32 <index>)
/*                       */ ///
/*                       */ /// The first argument is a pointer to a global variable containing the name of the entity
/*                       */ /// being instrumented. This should generally be the (mangled) function name for a set of
/*                       */ /// counters.
/*                       */ ///
/*                       */ /// The second argument is a hash value that can be used by the consumer of the profile data
/*                       */ /// to detect changes to the instrumented source, and the third is the number of counters
/*                       */ /// associated with name. It is an error if hash or num-counters differ between two
/*                       */ /// instances of instrprof.increment that refer to the same name.
/*                       */ ///
/*                       */ /// The last argument refers to which of the counters for name should be incremented. It
/*                       */ /// should be a value between 0 and num-counters.
/*                       */ ///
/*                       */ /// # Arguments
/*                       */ ///
/*                       */ /// `mangled_fn_name` - &'static ref to computed and injected static str, using:
/*                       */ ///
/*                       */ ///     ```
/*                       */ ///     fn rustc_symbol_mangling::compute_symbol_name(
/*                       */ ///         tcx: TyCtxt<'tcx>,
/*                       */ ///         instance: Instance<'tcx>,
/*                       */ ///         compute_instantiating_crate: impl FnOnce() -> CrateNum,
/*                       */ ///     ) -> String
/*                       */ ///     ```
/*                       */ ///
/*                       */ /// `source_version_hash` - Compute hash based that only changes if there are "significant"
/*                       */ /// to control-flow inside the function.
/*                       */ ///
/*                       */ /// `num_counters` - The total number of counter calls [MAX(counter_index) + 1] within the
/*                       */ /// function.
/*                       */ ///
/*                       */ /// `counter_index` - zero-based counter index scoped by the function. (Ordering of
/*                       */ /// counters, relative to the source code location, is apparently not expected.)
/*                       */ ///
/*                       */ /// # Notes
/*                       */ ///
/*                       */ /// * The mangled_fn_name may not be computable until generics are monomorphized (see
/*                       */ ///   parameters required by rustc_symbol_mangling::compute_symbol_name).
/*                       */ /// * The version hash may be computable from AST analysis, and may not benefit from further
/*                       */ ///   lowering.
/*                       */ /// * num_counters depends on having already identified all counter insertion locations.
/*                       */ /// * counter_index can be computed at time of counter insertion (incrementally).
/*                       */ /// * Numeric parameters are signed to match the llvm increment intrinsic parameter types.
/*                       */ fn __lower_incr_cov(_mangled_fn_name: &'static str, _fn_version_hash: i64, _num_counters: i32, _counter_index: i32) {
/*                       */ }
/*                       */
/*                       */ /// A coverage counter implementation that will work as both an intermediate coverage
/*                       */ /// counting and reporting implementation at the AST-level only--for debugging and
/*                       */ /// development--but also serves as a "marker" to be replaced by calls to LLVM
/*                       */ /// intrinsic coverage counter APIs during the lowering process.
/*                       */ ///
/*                       */ /// Calls to this function will be injected automatically into the AST. When LLVM intrinsics
/*                       */ /// are enabled, the counter function calls that were injected into the AST serve as
/*                       */ /// placeholders, to be replaced by an alternative, such as:
/*                       */ ///
/*                       */ ///     * direct invocation of the `llvm.instrprof.increment()` intrinsic; or
/*                       */ ///     * the `__lower_incr_cov()` function, defined above, that would invoke the
/*                       */ ///       `llvm.instrprof.increment()` intrinsic; or
/*                       */ ///     * a similar expression wrapper, with the additional parameters (as defined above
/*                       */ ///       for `__lower_incr_cov()`, that invokes `llvm.instrprof.increment()` and returns the
/*                       */ ///       result of the wrapped expression)
/*                       */ ///
/*                       */ /// The first two options would require replacing the inlined wrapper call with something
/*                       */ /// like:
/*                       */ ///
/*                       */ /// ```
/*                       */ /// { let result = {expr}; __inlined_incr_cov(context, counter); result }
/*                       */ /// ```
/*                       */ ///
/*                       */ /// But if the lowering process is already unwrapping the inlined call to `__incr_cov()`, then
/*                       */ /// it may be a perfect opportunity to replace the function with one of these more
/*                       */ /// direct methods.
/*                       */ ///
/*                       */ #[inline(always)]
/*                       */ pub fn __incr_cov(region_loc: &str, /*index: u32,*/) {
/*                       */     // Either call the intermediate non-llvm coverage counter API or
/*                       */     // replace the call to this function with the expanded `__lower_incr_cov()` call.
/*                       */
/*                       */     // let _lock = increment_counter(counter);
/*                       */     println!("{}", region_loc);
/*                       */ }
/*                       */
/*                       */ /// Write a report identifying each incremented counter and the number of times each counter
/*                       */ /// was incremented.
/*                       */ fn __report() {
/*                       */     println!("WRITE REPORT!");
/*                       */ }
/*                       */
/*                       */ /// Increment the counter after evaluating the wrapped expression (see `__incr_cov()`), then
/*                       */ /// write a report identifying each incremented counter and the number of times each counter
/*                       */ /// was incremented.
/*                       */ #[inline(always)]
/*                       */ pub fn __incr_cov_and_report<T>(region_loc: &str, /*counter: u32,*/ result: T) -> T {
/*                       */     __incr_cov(region_loc, /*counter,*/);
/*                       */     __report();
/*                       */     result
/*                       */ }
/*                       */
/*                       */ macro_rules! from {
/*                       */     ($from:expr) => { &format!("from: {}\n  to: {}:{}:{}", $from, file!(), line!(), column!()) };
/*                       */ }
/*                       */
/*                       */ macro_rules! to {
/*                       */     ($to:expr) => { &format!("to: {}\n  to: {}:{}:{}", $to, file!(), line!(), column!()) };
/*                       */ }
/*                       */
/*                       */ #[derive(Debug)]
/*                       */ enum TestEnum {
/*                       */     Red,
/*                       */     Green,
/*                       */     Blue,
/*                       */ }
/*                       */
/*                       */ struct TestStruct {
/*                       */     field: i32,
/*                       */ }
/*                       */
/*                       */ // IMPORTANT! IS WRAPPING main() ENOUGH? OR DO I ALSO NEED TO WRAP THREAD FUNCTIONS, ASSUMING
/*                       */ // THEY ARE STILL RUNNING WITH MAIN EXITS? (IF THEY CAN). NOT SURE HOW RUST HANDLES THAT.
/*                       */
/*                       */ // I SUSPECT USING THREAD_LOCAL COUNTERS MAY NOT ACTUALLY BE AN OPTIMIZATION OVER MUTEX LOCKS,
/*                       */ // BUT MAYBE I SHOULD ASK.
/*                       */
/*                       */ impl TestStruct {
/*    -                  */     fn new() -> Self {
/*    ┃                  */         __incr_cov(to!("end of fn new()")); // function-scoped counter index = 0
/*    ┃                  */         Self::new_with_value(31415)
/*    -                  */     }
/*                       */
/*    -                  */     fn new_with_value(field: i32) -> Self {
/*    ┃                  */         __incr_cov(to!("end of fn new_with_value()")); // function-scoped counter index = 0
/*    ┃                  */         Self {
/*    ┃                  */             field,
/*    ┃                  */         }
/*    -                  */     }
/*                       */
/*                       */     fn call_closure<F>(&self, closure: F) -> bool
/*                       */     where
/*                       */         F: FnOnce(
/*                       */             i32,
/*                       */         ) -> bool,
/*    -                  */     {
/*    ┃                  */         __incr_cov(to!("end of fn call_closure()")); // function-scoped counter index = 0
/*    ┃                  */         closure(123)
/*    -                  */     }
/*                       */
/*    -                  */     fn various(&self) -> Result<(),Error> {
/*    ┃                  */         __incr_cov(to!("just before next branch: after `match color`: pattern selection"));
/*    ┃                  */         use TestEnum::*;
/*    ┃                  */         let mut color = Red;
/*    ┃                  */         let _ = color;
/*    ┃                  */         color = Blue;
/*    ┃                  */         let _ = color;
/*    ┃                  */         color = Green;
/*    ┃                  */         match color { // function-scoped counter index = 0
/*    :                  */
/*    :                  */             // !!! RECORD SPAN FROM START OF INNERMOST CONTAINING BLOCK (THE FUNCTION IN THIS CASE) TO END OF MATCH EXPRESSION
/*    :                  */             // If `match`, `while`, `loop`, `for`, `if`, etc. expression has a `return`, `break`, or `continue`
/*    :                  */             // (if legal), then RECORD SPAN FROM START OF INNERMOST CONTAINING BLOCK TO END OF `return` EXPRESSION
/*    :                  */             // If the expression includes lazy booleans, nest calls to `__incr_cov()`.
/*    :   -              */             Red => {
/*    :   ┃              */                 __incr_cov(to!("end of matched Red"));
/*    :   ┃              */                 println!("roses");
/*    :   -              */             }
/*    :   -              */             Green => {
/*    :   ┃              */                 __incr_cov(to!("just before next branch: after `if spidey > goblin`"));
/*    :   ┃              */                 let spidey = 100;
/*    :   ┃              */                 let goblin = 50;
/*    :   ┃              */                 // if spidey > goblin {__incr_cov(from!(""),{
/*    :   ┃              */                 //     println!("what ev");
/*    :   ┃              */                 // })}
/*    :   ┃              */                 // ACTUALLY, WRAPPING THE ENTIRE IF BLOCK IN `__incr_cov` IS NOT A GREAT GENERAL RULE.
/*    :   ┃              */                 // JUST INSERTING A `return`, `break`, or `continue` IN THAT BLOCK (without an intermediate condition)
/*    :   ┃              */                 // MAKES THE `__incr_cov()` CALL UNREACHABLE!
/*    :   ┃              */                 // MY ORIGINAL SOLUTION WORKS BETTER (WRAP LAST EXPRESSION OR AFTER LAST SEMICOLON STATEMENT IN BLOCK)
/*    :   ┃              */                 // UNLESS THE EXPRESSION IS NOT A BLOCK.
/*    :   ┃   -          */                 if spidey > goblin {
/*    :   :   ┃          */                     __incr_cov(to!("end of if block, if no earlier branch in this scope"));
/*    :   :   ┃          */                     println!("spidey beats goblin");
/*    :   :   ┃          */
/*    :   ┃   -          */                 } else if {
/*    :   :   :          */                     // Make sure we can't compute the coverage count here.
/*    :   :   :          */                     // We know the expression executed if the previous if block DID NOT
/*    :   :   :          */                     // execute, and either this `else if` block does execute OR any subsequent
/*    :   :   :          */                     // `else if` or `else` blocks execute, OR none of the blocks in the
/*    :   :   :          */                     // `if`, `else if` or `else` blocks execute.
/*    :   :   :          */                     // `if`, `else if` or `else` blocks execute.
/*    :   :   ┃          */                     __incr_cov(to!("end of `else if spidey == goblin` expression"));
/*    :   :   ┃          */                     spidey == goblin
/*    :   ┃   -          */                 } {
/*    :   :   ┃          */                     __incr_cov(to!("end of if block, if no earlier branch in this scope"));
/*    :   :   ┃          */                     // COVERAGE NOTE: Do we mark only the expression span (that may be trivial, as in this case),
/*    :   :   ┃          */                     // or associate it with the outer block, similar to how the `if` expression is associated with
/*    :   :   ┃          */                     // the outer block? (Although it is a continuation, in a sense, it is discontiguous in this case,
/*    :   :   ┃          */                     // so I think simpler to just make it its own coverage region.)
/*    :   :   ┃          */                     println!("it's a draw");
/*    :   :   ┃          */
/*    :   ┃   -   -   -  */                 } else if {
/*    :   :   ┃          */                         __incr_cov(to!("end of `if true`"));
/*    :   ┃   -   -   -  */                         if true {
/*    :   :       :   ┃  */                             __incr_cov(to!("end of `return Ok(())`"));
/*  ┏-:---:-------:---<  */                             return Ok(());
/*  V :   :       ┃   -  */                         } else {
/*    :   :       :   ┃  */                             // __incr_cov(to!("end of else block"));
/*    :   :       :   ┃  */                             // computed counter expression
/*    :   :       :   ┃  */                             false
/*    :   :       :   -  */                         }
/*    :   :   -   -   -  */                     } {
/*    :   :   ┃          */                     __incr_cov(to!("end of if block"));
/*    :   :   ┃          */                     println!("wierd science");
/*    :   ┃   -          */                 } else {
/*    :   :   ┃          */                     // __incr_cov(to!("end of `return Ok(())"));
/*    :   :   ┃          */                     // counter expression: (start of Green match arm) - (if spidey > goblin) - (previous `} else if {`)
/*    :   :   ┃          */                     println!("goblin wins");
/*  ┏-:---:---<          */                     return Ok(()); // THIS COUNTS LAST STATEMENT IN `else` BLOCK
/*  V :   :   :          */                     // COVERAGE NOTE: When counting the span for `return`,
/*    :   :   :          */                     // `break`, or `continue`, also report the outer spans
/*    :   :   :          */                     // got this far--including this `else` block. Record
/*    :   :   :          */                     // The start positions for those outer blocks, but:
/*    :   :   :          */                     // * For the block containing the `return`, `break`, or
/*    :   :   :          */                     //   `continue`, end report the end position is the
/*    :   :   :          */                     //   start of the `return` span (or 1 char before it).
/*    :   :   :          */                     // * Anything else?
/*    :   ┃   -          */                 }
/*    :   :              */                 // __incr_cov(to!("end of matched Green"));
/*    :   :              */                 //  // DO NOT COUNT HERE IF NO STATEMENTS AFTER LAST `if` or `match`
/*    :   -              */             },
/*    :   -              */             Blue => {
/*    :   ┃              */                 __incr_cov(to!("end of matched Blue"));
/*    :   ┃              */                 println!("violets");
/*    :   -              */             }
/*    ┃                  */         }
/*    ┃                  */         __incr_cov(to!("just before next branch: after `if condition1` (HIR: 'match condition1')"));
/*    ┃                  */
/*    ┃                  */         let condition1 = true;
/*    ┃                  */         let condition2 = false;
/*    ┃                  */         let condition3 = true;
/*    ┃                  */
/*    ┃                  */         println!("Called `various()` for TestStruct with field={}", self.field);
/*    ┃                  */
/*    ┃   -              */         if condition1 {
/*    :   ┃              */             println!("before while loop");
/*    :   ┃              */             let mut countdown = 10;
/*    :   ┃              */              // Must increment before repeated while text expression
/*    :   :   I          */             while  countdown > 0 { // span is just the while test expression
/*    :   :   ┃          */                 println!("top of `while` loop");
/*    :   :   ┃          */                 countdown -= 1;
/*    :   :   ┃          */                 //  // Counter not needed, but span is computed as "while test" minus "block start"
/*    :   :   ┃          */                                                        // If test expression is 11, and the outer block runs only once, 11-1 = 10
/*    :   ┃   -          */             }
/*    :   ┃              */             println!("before for loop");
/*    :   ┃   -          */             for index in 0..10 {
/*    :   :   ┃          */                 println!("top of `for` loop");
/*    :   :   ┃   -      */                 if index == 8 {
/*    :   :   :   ┃      */                     println!("before break");
/*    :   :   :   ┃      */                     // note the following is not legal here:
/*    :   :   :   ┃      */                     //   "can only break with a value inside `loop` or breakable block"
/*    :   :   :   ┃      */                     // break
/*    :   :   :   ┃      */
/*    :   : ┏-----<      */                     break;
/*    :   : V :   :      */
/*    :   :   :   :      */                     // FIXME(richkadel): add examples with loop labels, breaking out of inner and outer loop to outer loop label, with expression.
/*    :   :   :   :      */                     // May want to record both the span and the start position after the broken out block depdnding on label
/*    :   :   ┃   -      */                 }
/*    :   :   ┃          */                 println!("after `break` test");
/*    :   :   ┃   -      */                 if condition2 {
/*  ┏-:---:---:---<      */                     return Ok(());
/*  V :   :   ┃   -      */                 }
/*    :   :   ┃          */
/*    :   :   ┃          */                 // BECAUSE THE PREVIOUS COVERAGE REGION HAS A `return`, THEN
/*    :   :   ┃          */                 // IF PREVIOUS COVERAGE REGION IS NOT COUNTED THEN OUTER REGION REACHED HERE.
/*    :   :   ┃          */                 // ADD A COVERAGE REGION FOR THE SPAN FROM JUST AFTER PREVIOUS REGION TO END
/*    :   :   ┃          */                 // OF OUTER SPAN, THEN TRUNCATE TO NEXT REGION NOT REACHED.
/*    :   :   ┃   -      */                 if index % 3 == 2 { // NO __incr_cov() HERE BECAUSE NO STATEMENTS BETWEEN LAST CONDITIONAL BLOCK AND START OF THIS ONE
/*    :   : Λ :   ┃      */
/*    :   : ┗-----<      */                     continue;
/*    :   :   ┃   -      */                 }
/*    :   :   ┃          */                 println!("after `continue` test");
/*    :   :   ┃          */                 // maybe add a runtime flag for a possible `return` here?
/*    :   :   ┃          */
/*    :   ┃   -          */             }
/*    :   ┃              */             println!("after for loop");
/*    :   ┃              */             let result = if { // START OF NEW CONDITIONAL EXPRESSION. NEXT "GUARANTEED" COUNTER SHOULD COUNT FROM END OF LAST CONDITIONAL EXPRESSION
/*    :   ┃              */                               // A "GUARANTEED" COUNTER CALL IS ONE THAT WILL BE CALLED REGARDLESS OF OTHER CONDITIONS. THIS INCLUDES:
/*    :   ┃              */                               //   * A CONDITIONAL EXPRESSION THAT IS NOT A BLOCK (OR ANOTHER CONDITIONAL STATEMENT, WHICH WOULD CONTAIN A BLOCK)
/*    :   ┃              */                               //   * OR IF THE NEXT CONDITIONAL EXPRESSION IS A BLOCK OR CONDITIONAL STATEMENT, THEN THE FIRST "GUARANTEED" COUNTER IN THAT BLOCK
/*    :   ┃              */                               //   * END OF BLOCK IF THE BLOCK DOES NOT HAVE INNER CONDITIONAL EXPRESSIONS
/*    :   ┃              */                               //   * BRANCHING STATEMENTS (`return`, `break`, `continue`) BY EITHER WRAPPING THE BRANCH STATEMENT NON-BLOCK EXPRESSION,
/*    :   ┃              */                               //     OR PREPENDING A COUNTER WITH EMPTY TUPLE IF NO EXPRESSION, OR IF EXPRESSION IS A BLOCK, THEN THE NEXT "GUARANTEED"
/*    :   ┃              */                               //     COUNTER CALL WITHIN THAT BLOCK.
/*    :   ┃              */                               //   BASICALLY, CARRY THE START OF COVERAGE SPAN FORWARD UNTIL THE GUARANTEED COUNTER IS FOUND
/*    :   ┃              */                 println!("after result = if ...");
/*    :   ┃       -      */                 if condition2 {
/*    :   :       ┃      */                     println!("before first return");
/*  ┏-:---:-------<      */                     return Ok(());
/*  V :   :       -      */                 } else if condition3 {
/*    :   :       ┃      */                     // THE ABOVE COUNTER IS _NOT_ REALLY NECESSARY IF EXPRESSION IS GUARANTEED TO EXECUTE.
/*    :   :       ┃      */                     // IF WE GET COUNTER IN `else if` BLOCK WE COVERED EXPRESSION.
/*    :   :       ┃      */                     // IF WE GET TO ANY REMAINING `else` or `else if` BLOCK WE KNOW WE EVALUATED THIS CONDITION
/*    :   :       ┃      */                     // AND ALL OTHERS UP TO THE EXECUTED BLOCK. BUT THE SPAN WOULD HAVE "HOLES" FOR UNEXECUTED BLOCKS.
/*    :   :       ┃      */                     println!("not second return");
/*  ┏-:---:-------<      */                     return Ok(());
/*  V :   :       -      */                 } else {
/*    :   :       ┃      */                     println!("not returning");
/*    :   :       ┃      */                     false
/*    :   :       -      */                 }
/*    :   ┃              */                 // NO COUNTER HERE BECAUSE NO STATEMENTS AFTER CONDITIONAL BLOCK
/*    :   ┃   -          */             } {
/*    :   :   ┃          */                 println!("branched condition returned true");
/*    :   :   ┃          */                 Ok(())
/*    :   ┃   -          */             } else if self.call_closure(
/*    :   :       -      */                     |closure_param|
/*    :   :       ┃   -  */                         if condition3 {
/*    :   :       :   ┃  */                             println!("in closure, captured condition said to print the param {}", closure_param);
/*    :   :       :   ┃  */                             false
/*    :   :       ┃   -  */                         } else {
/*    :   :       :   ┃  */                             println!("in closure, captured condition was false");
/*    :   :       :   ┃  */                             true
/*    :   :       ┃   -  */                         }
/*    :   :       -      */
/*    :   :   -          */                 ) {
/*    :   :   ┃          */                 println!("closure returned true");
/*    :   :   ┃          */                 Err(Error::new(ErrorKind::Other, "Result is error if closure returned true"))
/*    :   ┃   -          */             } else {
/*    :   :   ┃          */                 println!("closure returned false");
/*    :   :   ┃          */                 Err(Error::new(ErrorKind::Other, "Result is error if closure returned false"))
/*    :   ┃   -          */             };
/*    :   ┃              */             println!("bottom of function might be skipped if early `return`");
/*    :   ┃              */             result
/*    ┃   -              */         } else {
/*    :   ┃              */             println!("skipping everything in `various()`");
/*    :   ┃              */             Ok(())
/*    ┃   -              */         }
/*    ┃   -              */         // 0 // DO NOT COUNT IF NO STATEMENTS AFTER CONDITIONAL BLOCK. ALL COVERAGE IS ALREADY COUNTED
/*    -                  */     }
/*                       */ }
/*                       */
/*    -                  */ fn main() -> Result<(), std::io::Error> {
/*    ┃                  */     //let mut status: u8 = 2;
/*    ┃                  */     let mut status: u8 = 1;
/*    :       -          */     let result = if status < 2 &&
/*    :       ┃          */             {
/*    :       ┃          */                 status -= 1;
/*    :       ┃          */                 status == 0
/*    :   -   -          */             } {
/*    :   ┃              */         let test_struct = TestStruct::new_with_value(100);
/*    :   ┃              */         let _ = test_struct.various();
/*  ┏-:---<              */         return __incr_cov_and_report(from!(""),Err(Error::new(ErrorKind::Other, format!("Error status {}", status))))
/*  V :   -              */     } else {
/*    :   ┃              */         let test_struct = TestStruct::new();
/*    :   ┃              */         test_struct.various()
/*    :   -              */     };
/*    ┃                  */     println!("done");
/*    ┃                  */     __incr_cov_and_report(from!(""),result) // function-scoped counter index = 0
/*    -                  */ }