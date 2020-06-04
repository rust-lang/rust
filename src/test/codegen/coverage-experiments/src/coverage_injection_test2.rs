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
/*                       */ pub fn __incr_cov(region_loc: &str) {
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
/*                       */ macro_rules! from {
/*                       */     ($from:expr) => { &format!("from: {}\n  to: {}:{}:{}", $from, file!(), line!(), column!()) };
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
/*    ┃                  */         let __result = Self::new_with_value(31415); // function-scoped counter index = 0
/*    ┃                  */         __incr_cov(from!("fn new()"));
/*    ┃                  */         __result
/*    -                  */     }
/*                       */
/*    -                  */     fn new_with_value(field: i32) -> Self {
/*    ┃                  */         let __result = Self {
/*    ┃                  */             field,
/*    ┃                  */         };
/*    ┃                  */         __incr_cov(from!("fn new_with_value()")); // function-scoped counter index = 0
/*    ┃                  */         __result
/*    -                  */     }
/*                       */
/*                       */     fn call_closure<F>(&self, closure: F) -> bool
/*                       */     where
/*                       */         F: FnOnce(
/*                       */             i32,
/*                       */         ) -> bool,
/*    -                  */     {
/*    ┃                  */         let __result = closure(123);
/*    ┃                  */         __incr_cov(from!("fn call_closure()")); // function-scoped counter index = 0
/*    ┃                  */         __result
/*    -                  */     }
/*                       */
/*    -                  */     fn various(&self) -> Result<(),Error> {
/*    ┃                  */         use TestEnum::*;
/*    ┃                  */         let mut color = Red;
/*    ┃                  */         let _ = color;
/*    ┃                  */         color = Blue;
/*    ┃                  */         let _ = color;
/*    ┃                  */         color = Green;
/*    ┃                  */         match { let __result = color; __incr_cov(from!("fn various")); __result } { // function-scoped counter index = 0
/*    :                  */
/*    :                  */             // !!! RECORD SPAN FROM START OF INNERMOST CONTAINING BLOCK (THE FUNCTION IN THIS CASE) TO END OF MATCH EXPRESSION
/*    :                  */             // If `match`, `while`, `loop`, `for`, `if`, etc. expression has a `return`, `break`, or `continue`
/*    :                  */             // (if legal), then RECORD SPAN FROM START OF INNERMOST CONTAINING BLOCK TO END OF `return` EXPRESSION
/*    :                  */             // If the expression includes lazy booleans, nest calls to `__incr_cov()`.
/*    :   I              */             Red => {println!("roses"); __incr_cov(from!("Red => or end of MatchArmGuard expression inside pattern, if any"));}
/*    :   -              */             Green => {
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
/*    :   ┃   -          */                 if { let __result = spidey > goblin; __incr_cov(from!("Green => or end of MatchArmGuard expression inside pattern, if any")); __result } {
/*    :   :   ┃          */                     println!("spidey beats goblin");
/*    :   :   ┃          */                     __incr_cov(from!("block start"));
/*    :   ┃   -          */                 } else if { let __result = spidey == goblin; __incr_cov(from!("`else if` on this line")); __result } {
/*    :   :   ┃          */                     // COVERAGE NOTE: Do we mark only the expression span (that may be trivial, as in this case),
/*    :   :   ┃          */                     // or associate it with the outer block, similar to how the `if` expression is associated with
/*    :   :   ┃          */                     // the outer block? (Although it is a continuation, in a sense, it is discontiguous in this case,
/*    :   :   ┃          */                     // so I think simpler to just make it its own coverage region.)
/*    :   :   ┃          */                     println!("it's a draw");
/*    :   :   ┃          */                     __incr_cov(from!("block start"));
/*    :   ┃   -   -   -  */                 } else if if { let __result = true; __incr_cov(from!("`else if` on this line")); __result } {
/*    :   :       :   ┃  */                             // return __incr_cov(from!("after `if true`"),Ok(()));
/*    :   :       :   ┃  */                             // ACTUALLY, BECAUSE OF `return`, WE DO NOT RECORD THE `if true` EVEN THOUGH WE COVERED IT.
/*    :   :       :   ┃  */                             // IN FACT, IF THIS NESTED CONDITIONAL IN A CONDITIONAL EXPRESSION WAS AN `if` (WITHOUT PRECEDING ELSE)
/*    :   :       :   ┃  */                             // WE WOULD NOT HAVE RECORDED THE COVERAGE OF STATEMENTS LEADING UP TO THE `if`, SO
/*    :   :       :   ┃  */                             // IT SHOULD BE:
/*  ┏-:---:-------:---<  */                             return { let __result = Ok(()); __incr_cov(from!("")); __result };
/*  V :   :       :   :  */                             // NOTE THE `from` STRING IS SAME FOR THE `else if`s `__incr_cov` AND THIS `return`.
/*    :   :       :   :  */                             // ONLY ONE OF THESE WILL EXECUTE, TO RECORD COVERAGE FROM THAT SPOT.
/*    :   :       ┃   -  */                         } else {
/*    :   :       :   I  */                             { let __result = false; __incr_cov(from!("`else`")); __result }
/*    :   :   -   -      */                         } {
/*    :   :   ┃          */                     println!("wierd science");
/*    :   :   ┃          */                     __incr_cov(from!("block start"));
/*    :   ┃   -          */                 } else {
/*    :   :   ┃          */                     println!("goblin wins");
/*  ┏-:---:---<          */                     return { let __result = Ok(()); __incr_cov(from!("`else`")); __result }; // THIS COUNTS LAST STATEMENT IN `else` BLOCK
/*  V :   :   :          */                     // COVERAGE NOTE: When counting the span for `return`,
/*    :   :   :          */                     // `break`, or `continue`, also report the outer spans
/*    :   :   :          */                     // got this far--including this `else` block. Record
/*    :   :   :          */                     // The start positions for those outer blocks, but:
/*    :   :   :          */                     // * For the block containing the `return`, `break`, or
/*    :   :   :          */                     //   `continue`, end report the end position is the
/*    :   :   :          */                     //   start of the `return` span (or 1 char before it).
/*    :   :   :          */                     // * Anything else?
/*    :   ┃   -          */                 }
/*    :   ┃   -          */                 // __incr_cov(from!("")); // DO NOT COUNT HERE IF NO STATEMENTS AFTER LAST `if` or `match`
/*    :   -              */             },
/*    :   I              */             Blue => { println!("violets"); __incr_cov(from!("Blue => or end of MatchArmGuard expression inside pattern, if any")); }
/*    ┃                  */         }
/*    ┃                  */
/*    ┃                  */         let condition1 = true;
/*    ┃                  */         let condition2 = false;
/*    ┃                  */         let condition3 = true;
/*    ┃                  */
/*    ┃                  */         println!("Called `various()` for TestStruct with field={}", self.field);
/*    ┃                  */
/*    ┃   -              */         if { let __result = condition1; __incr_cov(from!("after block end of prior `match` (or `if-else if-else`)")); __result } {
/*    :   ┃              */             println!("before for loop");
/*    :   ┃   -          */             for index in { let __result = 0..10; __incr_cov(from!("block start")); __result } {
/*    :   :   ┃          */                 println!("top of `for` loop");
/*    :   :   ┃   -      */                 if { let __result = index == 8; __incr_cov(from!("block start")); __result } {
/*    :   :   :   ┃      */                     println!("before break");
/*    :   :   :   ┃      */                     // note the following is not legal here:
/*    :   :   :   ┃      */                     //   "can only break with a value inside `loop` or breakable block"
/*    :   :   :   ┃      */                     // break __incr_cov(from!(""));
/*    :   :   :   ┃      */                     __incr_cov(from!("block start"));
/*    :   : ┏-----<      */                     break;
/*    :   : V :   :      */
/*    :   :   :   :      */                     // FIXME(richkadel): add examples with loop labels, breaking out of inner and outer loop to outer loop label, with expression.
/*    :   :   :   :      */                     // May want to record both the span and the start position after the broken out block depdnding on label
/*    :   :   ┃   -      */                 }
/*    :   :   ┃          */                 println!("after `break` test");
/*    :   :   ┃   -      */                 if { let __result = condition2; __incr_cov(from!("block end of `if index == 8`")); __result } {
/*  ┏-:---:---:---<      */                     return { let __result = Ok(()); __incr_cov(from!("block start")); __result };
/*  V :   :   ┃   -      */                 }
/*    :   :   ┃          */
/*    :   :   ┃          */                 // BECAUSE THE PREVIOUS COVERAGE REGION HAS A `return`, THEN
/*    :   :   ┃          */                 // IF PREVIOUS COVERAGE REGION IS NOT COUNTED THEN OUTER REGION REACHED HERE.
/*    :   :   ┃          */                 // ADD A COVERAGE REGION FOR THE SPAN FROM JUST AFTER PREVIOUS REGION TO END
/*    :   :   ┃          */                 // OF OUTER SPAN, THEN TRUNCATE TO NEXT REGION NOT REACHED.
/*    :   :   ┃   -      */                 if index % 3 == 2 { // NO __incr_cov() HERE BECAUSE NO STATEMENTS BETWEEN LAST CONDITIONAL BLOCK AND START OF THIS ONE
/*    :   : Λ :   ┃      */                     __incr_cov(from!("block end of `if condition2`"));
/*    :   : ┗-----<      */                     continue;
/*    :   :   ┃   -      */                 }
/*    :   :   ┃          */                 println!("after `continue` test");
/*    :   :   ┃          */                 // maybe add a runtime flag for a possible `return` here?
/*    :   :   ┃          */                 __incr_cov(from!(""));
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
/*    :   ┃       -      */                 if { let __result = condition2; __incr_cov(from!("block end of `for` loop")); __result } {
/*    :   :       ┃      */                     println!("before first return");
/*  ┏-:---:-------<      */                     return { let __result = Ok(()); __incr_cov(from!("block start")); __result };
/*  V :   :       -      */                 } else if { let __result = condition3; __incr_cov(from!("`else`")); __result } {
/*    :   :       ┃      */                     // THE ABOVE COUNTER IS _NOT_ REALLY NECESSARY IF EXPRESSION IS GUARANTEED TO EXECUTE.
/*    :   :       ┃      */                     // IF WE GET COUNTER IN `else if` BLOCK WE COVERED EXPRESSION.
/*    :   :       ┃      */                     // IF WE GET TO ANY REMAINING `else` or `else if` BLOCK WE KNOW WE EVALUATED THIS CONDITION
/*    :   :       ┃      */                     // AND ALL OTHERS UP TO THE EXECUTED BLOCK. BUT THE SPAN WOULD HAVE "HOLES" FOR UNEXECUTED BLOCKS.
/*    :   :       ┃      */                     println!("not second return");
/*  ┏-:---:-------<      */                     return { let __result = Ok(()); __incr_cov(from!("block start")); __result };
/*  V :   :       -      */                 } else {
/*    :   :       ┃      */                     println!("not returning");
/*    :   :       ┃      */                     { let __result = false; __incr_cov(from!("block start")); __result }
/*    :   :       -      */                 }
/*    :   ┃              */                 // NO COUNTER HERE BECAUSE NO STATEMENTS AFTER CONDITIONAL BLOCK
/*    :   ┃   -          */             } {
/*    :   :   ┃          */                 println!("branched condition returned true");
/*    :   :   ┃          */                 { let __result = Ok(()); __incr_cov(from!("")); __result }
/*    :   ┃   -          */             } else if self.call_closure(
/*    :   :       -      */                     |closure_param| {
/*    :   :       ┃   -  */                         let __result = if condition3 {
/*    :   :       :   ┃  */                             println!("in closure, captured condition said to print the param {}", closure_param);
/*    :   :       :   ┃  */                             { let __result = false; __incr_cov(from!("")); __result }
/*    :   :       ┃   -  */                         } else {
/*    :   :       :   ┃  */                             println!("in closure, captured condition was false");
/*    :   :       :   ┃  */                             { let __result = true; __incr_cov(from!("")); __result }
/*    :   :       ┃   -  */                         };
/*    :   :       -      */                         __incr_cov(from!("")); __result }
/*    :   :   -          */                 ) {
/*    :   :   ┃          */                 println!("closure returned true");
/*    :   :   ┃          */                 { let __result = Err(Error::new(ErrorKind::Other, "Result is error if closure returned true")); __incr_cov(from!("")); __result }
/*    :   ┃   -          */             } else {
/*    :   :   ┃          */                 println!("closure returned false");
/*    :   :   ┃          */                 { let __result = Err(Error::new(ErrorKind::Other, "Result is error if closure returned false")); __incr_cov(from!("")); __result }
/*    :   ┃   -          */             };
/*    :   ┃              */             println!("bottom of function might be skipped if early `return`");
/*    :   ┃              */             { let __result = result; __incr_cov(from!("if condition1")); __result }
/*    ┃   -              */         } else {
/*    :   ┃              */             println!("skipping everything in `various()`");
/*    :   ┃              */             { let __result = Ok(()); __incr_cov(from!("")); __result }
/*    ┃   -              */         }
/*    ┃   -              */         // __incr_cov(from!(""),0) // DO NOT COUNT IF NO STATEMENTS AFTER CONDITIONAL BLOCK. ALL COVERAGE IS ALREADY COUNTED
/*    -                  */     }
/*                       */ }
/*                       */
/*    -                  */ fn main() -> Result<(), std::io::Error> {
/*    ┃                  */     //let mut status: u8 = 2;
/*    ┃                  */     let mut status: u8 = 1;
/*    :       -          */     let result = if status < 2 &&
/*    :       ┃          */             { let __result = {
/*    :       ┃          */                 status -= 1;
/*    :       ┃          */                 status == 0
/*    :   -   -          */             }; __incr_cov(from!("")); __result } {
/*    :   ┃              */         let test_struct = TestStruct::new_with_value(100);
/*    :   ┃              */         let _ = test_struct.various();
/*  ┏-:---<              */         return { let __result = Err(Error::new(ErrorKind::Other, format!("Error status {}", status))); __incr_cov(from!("")); __report(); __result }
/*  V :   -              */     } else {
/*    :   ┃              */         let test_struct = TestStruct::new();
/*    :   ┃              */         { let __result = test_struct.various(); __incr_cov(from!("")); __result }
/*    :   -              */     };
/*    ┃                  */     println!("done");
/*    ┃                  */     { let __result = result; __incr_cov(from!("")); __report(); __result }
/*    -                  */ }