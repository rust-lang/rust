mod explicit_counter_loop;
mod for_loop_arg;
mod for_loop_over_map_kv;
mod for_loop_range;
mod for_mut_range_bound;
mod for_single_element_loop;
mod infinite_loop;
mod manual_flatten;
mod manual_memcpy;
mod needless_collect;
mod never_loop;
mod same_item_push;
mod utils;
mod while_let_on_iterator;

use crate::utils::sugg::Sugg;
use crate::utils::{
    higher, is_in_panic_handler, is_no_std_crate, snippet_with_applicability, span_lint_and_help, span_lint_and_sugg,
    sugg,
};
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, LoopSource, MatchSource, Pat, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use utils::{get_span_of_entire_for_loop, make_iterator_snippet, IncrementVisitor, InitializeVisitor};

declare_clippy_lint! {
    /// **What it does:** Checks for for-loops that manually copy items between
    /// slices that could be optimized by having a memcpy.
    ///
    /// **Why is this bad?** It is not as fast as a memcpy.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # let src = vec![1];
    /// # let mut dst = vec![0; 65];
    /// for i in 0..src.len() {
    ///     dst[i + 64] = src[i];
    /// }
    /// ```
    /// Could be written as:
    /// ```rust
    /// # let src = vec![1];
    /// # let mut dst = vec![0; 65];
    /// dst[64..(src.len() + 64)].clone_from_slice(&src[..]);
    /// ```
    pub MANUAL_MEMCPY,
    perf,
    "manually copying items between slices"
}

declare_clippy_lint! {
    /// **What it does:** Checks for looping over the range of `0..len` of some
    /// collection just to get the values by index.
    ///
    /// **Why is this bad?** Just iterating the collection itself makes the intent
    /// more clear and is probably faster.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let vec = vec!['a', 'b', 'c'];
    /// for i in 0..vec.len() {
    ///     println!("{}", vec[i]);
    /// }
    /// ```
    /// Could be written as:
    /// ```rust
    /// let vec = vec!['a', 'b', 'c'];
    /// for i in vec {
    ///     println!("{}", i);
    /// }
    /// ```
    pub NEEDLESS_RANGE_LOOP,
    style,
    "for-looping over a range of indices where an iterator over items would do"
}

declare_clippy_lint! {
    /// **What it does:** Checks for loops on `x.iter()` where `&x` will do, and
    /// suggests the latter.
    ///
    /// **Why is this bad?** Readability.
    ///
    /// **Known problems:** False negatives. We currently only warn on some known
    /// types.
    ///
    /// **Example:**
    /// ```rust
    /// // with `y` a `Vec` or slice:
    /// # let y = vec![1];
    /// for x in y.iter() {
    ///     // ..
    /// }
    /// ```
    /// can be rewritten to
    /// ```rust
    /// # let y = vec![1];
    /// for x in &y {
    ///     // ..
    /// }
    /// ```
    pub EXPLICIT_ITER_LOOP,
    pedantic,
    "for-looping over `_.iter()` or `_.iter_mut()` when `&_` or `&mut _` would do"
}

declare_clippy_lint! {
    /// **What it does:** Checks for loops on `y.into_iter()` where `y` will do, and
    /// suggests the latter.
    ///
    /// **Why is this bad?** Readability.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```rust
    /// # let y = vec![1];
    /// // with `y` a `Vec` or slice:
    /// for x in y.into_iter() {
    ///     // ..
    /// }
    /// ```
    /// can be rewritten to
    /// ```rust
    /// # let y = vec![1];
    /// for x in y {
    ///     // ..
    /// }
    /// ```
    pub EXPLICIT_INTO_ITER_LOOP,
    pedantic,
    "for-looping over `_.into_iter()` when `_` would do"
}

declare_clippy_lint! {
    /// **What it does:** Checks for loops on `x.next()`.
    ///
    /// **Why is this bad?** `next()` returns either `Some(value)` if there was a
    /// value, or `None` otherwise. The insidious thing is that `Option<_>`
    /// implements `IntoIterator`, so that possibly one value will be iterated,
    /// leading to some hard to find bugs. No one will want to write such code
    /// [except to win an Underhanded Rust
    /// Contest](https://www.reddit.com/r/rust/comments/3hb0wm/underhanded_rust_contest/cu5yuhr).
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// for x in y.next() {
    ///     ..
    /// }
    /// ```
    pub ITER_NEXT_LOOP,
    correctness,
    "for-looping over `_.next()` which is probably not intended"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `for` loops over `Option` or `Result` values.
    ///
    /// **Why is this bad?** Readability. This is more clearly expressed as an `if
    /// let`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # let opt = Some(1);
    ///
    /// // Bad
    /// for x in opt {
    ///     // ..
    /// }
    ///
    /// // Good
    /// if let Some(x) = opt {
    ///     // ..
    /// }
    /// ```
    ///
    /// // or
    ///
    /// ```rust
    /// # let res: Result<i32, std::io::Error> = Ok(1);
    ///
    /// // Bad
    /// for x in &res {
    ///     // ..
    /// }
    ///
    /// // Good
    /// if let Ok(x) = res {
    ///     // ..
    /// }
    /// ```
    pub FOR_LOOPS_OVER_FALLIBLES,
    correctness,
    "for-looping over an `Option` or a `Result`, which is more clearly expressed as an `if let`"
}

declare_clippy_lint! {
    /// **What it does:** Detects `loop + match` combinations that are easier
    /// written as a `while let` loop.
    ///
    /// **Why is this bad?** The `while let` loop is usually shorter and more
    /// readable.
    ///
    /// **Known problems:** Sometimes the wrong binding is displayed ([#383](https://github.com/rust-lang/rust-clippy/issues/383)).
    ///
    /// **Example:**
    /// ```rust,no_run
    /// # let y = Some(1);
    /// loop {
    ///     let x = match y {
    ///         Some(x) => x,
    ///         None => break,
    ///     };
    ///     // .. do something with x
    /// }
    /// // is easier written as
    /// while let Some(x) = y {
    ///     // .. do something with x
    /// };
    /// ```
    pub WHILE_LET_LOOP,
    complexity,
    "`loop { if let { ... } else break }`, which can be written as a `while let` loop"
}

declare_clippy_lint! {
    /// **What it does:** Checks for functions collecting an iterator when collect
    /// is not needed.
    ///
    /// **Why is this bad?** `collect` causes the allocation of a new data structure,
    /// when this allocation may not be needed.
    ///
    /// **Known problems:**
    /// None
    ///
    /// **Example:**
    /// ```rust
    /// # let iterator = vec![1].into_iter();
    /// let len = iterator.clone().collect::<Vec<_>>().len();
    /// // should be
    /// let len = iterator.count();
    /// ```
    pub NEEDLESS_COLLECT,
    perf,
    "collecting an iterator when collect is not needed"
}

declare_clippy_lint! {
    /// **What it does:** Checks `for` loops over slices with an explicit counter
    /// and suggests the use of `.enumerate()`.
    ///
    /// **Why is it bad?** Using `.enumerate()` makes the intent more clear,
    /// declutters the code and may be faster in some instances.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # let v = vec![1];
    /// # fn bar(bar: usize, baz: usize) {}
    /// let mut i = 0;
    /// for item in &v {
    ///     bar(i, *item);
    ///     i += 1;
    /// }
    /// ```
    /// Could be written as
    /// ```rust
    /// # let v = vec![1];
    /// # fn bar(bar: usize, baz: usize) {}
    /// for (i, item) in v.iter().enumerate() { bar(i, *item); }
    /// ```
    pub EXPLICIT_COUNTER_LOOP,
    complexity,
    "for-looping with an explicit counter when `_.enumerate()` would do"
}

declare_clippy_lint! {
    /// **What it does:** Checks for empty `loop` expressions.
    ///
    /// **Why is this bad?** These busy loops burn CPU cycles without doing
    /// anything. It is _almost always_ a better idea to `panic!` than to have
    /// a busy loop.
    ///
    /// If panicking isn't possible, think of the environment and either:
    ///   - block on something
    ///   - sleep the thread for some microseconds
    ///   - yield or pause the thread
    ///
    /// For `std` targets, this can be done with
    /// [`std::thread::sleep`](https://doc.rust-lang.org/std/thread/fn.sleep.html)
    /// or [`std::thread::yield_now`](https://doc.rust-lang.org/std/thread/fn.yield_now.html).
    ///
    /// For `no_std` targets, doing this is more complicated, especially because
    /// `#[panic_handler]`s can't panic. To stop/pause the thread, you will
    /// probably need to invoke some target-specific intrinsic. Examples include:
    ///   - [`x86_64::instructions::hlt`](https://docs.rs/x86_64/0.12.2/x86_64/instructions/fn.hlt.html)
    ///   - [`cortex_m::asm::wfi`](https://docs.rs/cortex-m/0.6.3/cortex_m/asm/fn.wfi.html)
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```no_run
    /// loop {}
    /// ```
    pub EMPTY_LOOP,
    style,
    "empty `loop {}`, which should block or sleep"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `while let` expressions on iterators.
    ///
    /// **Why is this bad?** Readability. A simple `for` loop is shorter and conveys
    /// the intent better.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// while let Some(val) = iter() {
    ///     ..
    /// }
    /// ```
    pub WHILE_LET_ON_ITERATOR,
    style,
    "using a `while let` loop instead of a for loop on an iterator"
}

declare_clippy_lint! {
    /// **What it does:** Checks for iterating a map (`HashMap` or `BTreeMap`) and
    /// ignoring either the keys or values.
    ///
    /// **Why is this bad?** Readability. There are `keys` and `values` methods that
    /// can be used to express that don't need the values or keys.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// for (k, _) in &map {
    ///     ..
    /// }
    /// ```
    ///
    /// could be replaced by
    ///
    /// ```ignore
    /// for k in map.keys() {
    ///     ..
    /// }
    /// ```
    pub FOR_KV_MAP,
    style,
    "looping on a map using `iter` when `keys` or `values` would do"
}

declare_clippy_lint! {
    /// **What it does:** Checks for loops that will always `break`, `return` or
    /// `continue` an outer loop.
    ///
    /// **Why is this bad?** This loop never loops, all it does is obfuscating the
    /// code.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```rust
    /// loop {
    ///     ..;
    ///     break;
    /// }
    /// ```
    pub NEVER_LOOP,
    correctness,
    "any loop that will always `break` or `return`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for loops which have a range bound that is a mutable variable
    ///
    /// **Why is this bad?** One might think that modifying the mutable variable changes the loop bounds
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```rust
    /// let mut foo = 42;
    /// for i in 0..foo {
    ///     foo -= 1;
    ///     println!("{}", i); // prints numbers from 0 to 42, not 0 to 21
    /// }
    /// ```
    pub MUT_RANGE_BOUND,
    complexity,
    "for loop over a range where one of the bounds is a mutable variable"
}

declare_clippy_lint! {
    /// **What it does:** Checks whether variables used within while loop condition
    /// can be (and are) mutated in the body.
    ///
    /// **Why is this bad?** If the condition is unchanged, entering the body of the loop
    /// will lead to an infinite loop.
    ///
    /// **Known problems:** If the `while`-loop is in a closure, the check for mutation of the
    /// condition variables in the body can cause false negatives. For example when only `Upvar` `a` is
    /// in the condition and only `Upvar` `b` gets mutated in the body, the lint will not trigger.
    ///
    /// **Example:**
    /// ```rust
    /// let i = 0;
    /// while i > 10 {
    ///     println!("let me loop forever!");
    /// }
    /// ```
    pub WHILE_IMMUTABLE_CONDITION,
    correctness,
    "variables used within while expression are not mutated in the body"
}

declare_clippy_lint! {
    /// **What it does:** Checks whether a for loop is being used to push a constant
    /// value into a Vec.
    ///
    /// **Why is this bad?** This kind of operation can be expressed more succinctly with
    /// `vec![item;SIZE]` or `vec.resize(NEW_SIZE, item)` and using these alternatives may also
    /// have better performance.
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```rust
    /// let item1 = 2;
    /// let item2 = 3;
    /// let mut vec: Vec<u8> = Vec::new();
    /// for _ in 0..20 {
    ///    vec.push(item1);
    /// }
    /// for _ in 0..30 {
    ///     vec.push(item2);
    /// }
    /// ```
    /// could be written as
    /// ```rust
    /// let item1 = 2;
    /// let item2 = 3;
    /// let mut vec: Vec<u8> = vec![item1; 20];
    /// vec.resize(20 + 30, item2);
    /// ```
    pub SAME_ITEM_PUSH,
    style,
    "the same item is pushed inside of a for loop"
}

declare_clippy_lint! {
    /// **What it does:** Checks whether a for loop has a single element.
    ///
    /// **Why is this bad?** There is no reason to have a loop of a
    /// single element.
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```rust
    /// let item1 = 2;
    /// for item in &[item1] {
    ///     println!("{}", item);
    /// }
    /// ```
    /// could be written as
    /// ```rust
    /// let item1 = 2;
    /// let item = &item1;
    /// println!("{}", item);
    /// ```
    pub SINGLE_ELEMENT_LOOP,
    complexity,
    "there is no reason to have a single element loop"
}

declare_clippy_lint! {
    /// **What it does:** Check for unnecessary `if let` usage in a for loop
    /// where only the `Some` or `Ok` variant of the iterator element is used.
    ///
    /// **Why is this bad?** It is verbose and can be simplified
    /// by first calling the `flatten` method on the `Iterator`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let x = vec![Some(1), Some(2), Some(3)];
    /// for n in x {
    ///     if let Some(n) = n {
    ///         println!("{}", n);
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// let x = vec![Some(1), Some(2), Some(3)];
    /// for n in x.into_iter().flatten() {
    ///     println!("{}", n);
    /// }
    /// ```
    pub MANUAL_FLATTEN,
    complexity,
    "for loops over `Option`s or `Result`s with a single expression can be simplified"
}

declare_lint_pass!(Loops => [
    MANUAL_MEMCPY,
    MANUAL_FLATTEN,
    NEEDLESS_RANGE_LOOP,
    EXPLICIT_ITER_LOOP,
    EXPLICIT_INTO_ITER_LOOP,
    ITER_NEXT_LOOP,
    FOR_LOOPS_OVER_FALLIBLES,
    WHILE_LET_LOOP,
    NEEDLESS_COLLECT,
    EXPLICIT_COUNTER_LOOP,
    EMPTY_LOOP,
    WHILE_LET_ON_ITERATOR,
    FOR_KV_MAP,
    NEVER_LOOP,
    MUT_RANGE_BOUND,
    WHILE_IMMUTABLE_CONDITION,
    SAME_ITEM_PUSH,
    SINGLE_ELEMENT_LOOP,
]);

impl<'tcx> LateLintPass<'tcx> for Loops {
    #[allow(clippy::too_many_lines)]
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let Some((pat, arg, body, span)) = higher::for_loop(expr) {
            // we don't want to check expanded macros
            // this check is not at the top of the function
            // since higher::for_loop expressions are marked as expansions
            if body.span.from_expansion() {
                return;
            }
            check_for_loop(cx, pat, arg, body, expr, span);
        }

        // we don't want to check expanded macros
        if expr.span.from_expansion() {
            return;
        }

        // check for never_loop
        never_loop::check_never_loop(cx, expr);

        // check for `loop { if let {} else break }` that could be `while let`
        // (also matches an explicit "match" instead of "if let")
        // (even if the "match" or "if let" is used for declaration)
        if let ExprKind::Loop(ref block, _, LoopSource::Loop, _) = expr.kind {
            // also check for empty `loop {}` statements, skipping those in #[panic_handler]
            if block.stmts.is_empty() && block.expr.is_none() && !is_in_panic_handler(cx, expr) {
                let msg = "empty `loop {}` wastes CPU cycles";
                let help = if is_no_std_crate(cx.tcx.hir().krate()) {
                    "you should either use `panic!()` or add a call pausing or sleeping the thread to the loop body"
                } else {
                    "you should either use `panic!()` or add `std::thread::sleep(..);` to the loop body"
                };
                span_lint_and_help(cx, EMPTY_LOOP, expr.span, msg, None, help);
            }

            // extract the expression from the first statement (if any) in a block
            let inner_stmt_expr = extract_expr_from_first_stmt(block);
            // or extract the first expression (if any) from the block
            if let Some(inner) = inner_stmt_expr.or_else(|| extract_first_expr(block)) {
                if let ExprKind::Match(ref matchexpr, ref arms, ref source) = inner.kind {
                    // ensure "if let" compatible match structure
                    match *source {
                        MatchSource::Normal | MatchSource::IfLetDesugar { .. } => {
                            if arms.len() == 2
                                && arms[0].guard.is_none()
                                && arms[1].guard.is_none()
                                && is_simple_break_expr(&arms[1].body)
                            {
                                if in_external_macro(cx.sess(), expr.span) {
                                    return;
                                }

                                // NOTE: we used to build a body here instead of using
                                // ellipsis, this was removed because:
                                // 1) it was ugly with big bodies;
                                // 2) it was not indented properly;
                                // 3) it wasn’t very smart (see #675).
                                let mut applicability = Applicability::HasPlaceholders;
                                span_lint_and_sugg(
                                    cx,
                                    WHILE_LET_LOOP,
                                    expr.span,
                                    "this loop could be written as a `while let` loop",
                                    "try",
                                    format!(
                                        "while let {} = {} {{ .. }}",
                                        snippet_with_applicability(cx, arms[0].pat.span, "..", &mut applicability),
                                        snippet_with_applicability(cx, matchexpr.span, "..", &mut applicability),
                                    ),
                                    applicability,
                                );
                            }
                        },
                        _ => (),
                    }
                }
            }
        }

        while_let_on_iterator::check_while_let_on_iterator(cx, expr);

        if let Some((cond, body)) = higher::while_loop(&expr) {
            infinite_loop::check_infinite_loop(cx, cond, body);
        }

        needless_collect::check_needless_collect(expr, cx);
    }
}

fn check_for_loop<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    arg: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
    expr: &'tcx Expr<'_>,
    span: Span,
) {
    let is_manual_memcpy_triggered = manual_memcpy::detect_manual_memcpy(cx, pat, arg, body, expr);
    if !is_manual_memcpy_triggered {
        for_loop_range::check_for_loop_range(cx, pat, arg, body, expr);
        explicit_counter_loop::check_for_loop_explicit_counter(cx, pat, arg, body, expr);
    }
    for_loop_arg::check_for_loop_arg(cx, pat, arg, expr);
    for_loop_over_map_kv::check_for_loop_over_map_kv(cx, pat, arg, body, expr);
    for_mut_range_bound::check_for_mut_range_bound(cx, arg, body);
    for_single_element_loop::check_for_single_element_loop(cx, pat, arg, body, expr);
    same_item_push::detect_same_item_push(cx, pat, arg, body, expr);
    manual_flatten::check_manual_flatten(cx, pat, arg, body, span);
}

/// a wrapper of `Sugg`. Besides what `Sugg` do, this removes unnecessary `0`;
/// and also, it avoids subtracting a variable from the same one by replacing it with `0`.
/// it exists for the convenience of the overloaded operators while normal functions can do the
/// same.
#[derive(Clone)]
struct MinifyingSugg<'a>(Sugg<'a>);

impl<'a> MinifyingSugg<'a> {
    fn as_str(&self) -> &str {
        let Sugg::NonParen(s) | Sugg::MaybeParen(s) | Sugg::BinOp(_, s) = &self.0;
        s.as_ref()
    }

    fn into_sugg(self) -> Sugg<'a> {
        self.0
    }
}

impl<'a> From<Sugg<'a>> for MinifyingSugg<'a> {
    fn from(sugg: Sugg<'a>) -> Self {
        Self(sugg)
    }
}

impl std::ops::Add for &MinifyingSugg<'static> {
    type Output = MinifyingSugg<'static>;
    fn add(self, rhs: &MinifyingSugg<'static>) -> MinifyingSugg<'static> {
        match (self.as_str(), rhs.as_str()) {
            ("0", _) => rhs.clone(),
            (_, "0") => self.clone(),
            (_, _) => (&self.0 + &rhs.0).into(),
        }
    }
}

impl std::ops::Sub for &MinifyingSugg<'static> {
    type Output = MinifyingSugg<'static>;
    fn sub(self, rhs: &MinifyingSugg<'static>) -> MinifyingSugg<'static> {
        match (self.as_str(), rhs.as_str()) {
            (_, "0") => self.clone(),
            ("0", _) => (-rhs.0.clone()).into(),
            (x, y) if x == y => sugg::ZERO.into(),
            (_, _) => (&self.0 - &rhs.0).into(),
        }
    }
}

impl std::ops::Add<&MinifyingSugg<'static>> for MinifyingSugg<'static> {
    type Output = MinifyingSugg<'static>;
    fn add(self, rhs: &MinifyingSugg<'static>) -> MinifyingSugg<'static> {
        match (self.as_str(), rhs.as_str()) {
            ("0", _) => rhs.clone(),
            (_, "0") => self,
            (_, _) => (self.0 + &rhs.0).into(),
        }
    }
}

impl std::ops::Sub<&MinifyingSugg<'static>> for MinifyingSugg<'static> {
    type Output = MinifyingSugg<'static>;
    fn sub(self, rhs: &MinifyingSugg<'static>) -> MinifyingSugg<'static> {
        match (self.as_str(), rhs.as_str()) {
            (_, "0") => self,
            ("0", _) => (-rhs.0.clone()).into(),
            (x, y) if x == y => sugg::ZERO.into(),
            (_, _) => (self.0 - &rhs.0).into(),
        }
    }
}

/// If a block begins with a statement (possibly a `let` binding) and has an
/// expression, return it.
fn extract_expr_from_first_stmt<'tcx>(block: &Block<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if block.stmts.is_empty() {
        return None;
    }
    if let StmtKind::Local(ref local) = block.stmts[0].kind {
        local.init //.map(|expr| expr)
    } else {
        None
    }
}

/// If a block begins with an expression (with or without semicolon), return it.
fn extract_first_expr<'tcx>(block: &Block<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    match block.expr {
        Some(ref expr) if block.stmts.is_empty() => Some(expr),
        None if !block.stmts.is_empty() => match block.stmts[0].kind {
            StmtKind::Expr(ref expr) | StmtKind::Semi(ref expr) => Some(expr),
            StmtKind::Local(..) | StmtKind::Item(..) => None,
        },
        _ => None,
    }
}

/// Returns `true` if expr contains a single break expr without destination label
/// and
/// passed expression. The expression may be within a block.
fn is_simple_break_expr(expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Break(dest, ref passed_expr) if dest.label.is_none() && passed_expr.is_none() => true,
        ExprKind::Block(ref b, _) => extract_first_expr(b).map_or(false, |subexpr| is_simple_break_expr(subexpr)),
        _ => false,
    }
}
