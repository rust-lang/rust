mod empty_loop;
mod explicit_counter_loop;
mod explicit_into_iter_loop;
mod explicit_iter_loop;
mod for_kv_map;
mod for_loops_over_fallibles;
mod iter_next_loop;
mod manual_flatten;
mod manual_memcpy;
mod mut_range_bound;
mod needless_collect;
mod needless_range_loop;
mod never_loop;
mod same_item_push;
mod single_element_loop;
mod utils;
mod while_immutable_condition;
mod while_let_loop;
mod while_let_on_iterator;

use clippy_utils::higher;
use rustc_hir::{Expr, ExprKind, LoopSource, Pat};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use utils::{make_iterator_snippet, IncrementVisitor, InitializeVisitor};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for for-loops that manually copy items between
    /// slices that could be optimized by having a memcpy.
    ///
    /// ### Why is this bad?
    /// It is not as fast as a memcpy.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for looping over the range of `0..len` of some
    /// collection just to get the values by index.
    ///
    /// ### Why is this bad?
    /// Just iterating the collection itself makes the intent
    /// more clear and is probably faster.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for loops on `x.iter()` where `&x` will do, and
    /// suggests the latter.
    ///
    /// ### Why is this bad?
    /// Readability.
    ///
    /// ### Known problems
    /// False negatives. We currently only warn on some known
    /// types.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for loops on `y.into_iter()` where `y` will do, and
    /// suggests the latter.
    ///
    /// ### Why is this bad?
    /// Readability.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for loops on `x.next()`.
    ///
    /// ### Why is this bad?
    /// `next()` returns either `Some(value)` if there was a
    /// value, or `None` otherwise. The insidious thing is that `Option<_>`
    /// implements `IntoIterator`, so that possibly one value will be iterated,
    /// leading to some hard to find bugs. No one will want to write such code
    /// [except to win an Underhanded Rust
    /// Contest](https://www.reddit.com/r/rust/comments/3hb0wm/underhanded_rust_contest/cu5yuhr).
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for `for` loops over `Option` or `Result` values.
    ///
    /// ### Why is this bad?
    /// Readability. This is more clearly expressed as an `if
    /// let`.
    ///
    /// ### Example
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
    suspicious,
    "for-looping over an `Option` or a `Result`, which is more clearly expressed as an `if let`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detects `loop + match` combinations that are easier
    /// written as a `while let` loop.
    ///
    /// ### Why is this bad?
    /// The `while let` loop is usually shorter and more
    /// readable.
    ///
    /// ### Known problems
    /// Sometimes the wrong binding is displayed ([#383](https://github.com/rust-lang/rust-clippy/issues/383)).
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for functions collecting an iterator when collect
    /// is not needed.
    ///
    /// ### Why is this bad?
    /// `collect` causes the allocation of a new data structure,
    /// when this allocation may not be needed.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks `for` loops over slices with an explicit counter
    /// and suggests the use of `.enumerate()`.
    ///
    /// ### Why is this bad?
    /// Using `.enumerate()` makes the intent more clear,
    /// declutters the code and may be faster in some instances.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for empty `loop` expressions.
    ///
    /// ### Why is this bad?
    /// These busy loops burn CPU cycles without doing
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
    /// ### Example
    /// ```no_run
    /// loop {}
    /// ```
    pub EMPTY_LOOP,
    suspicious,
    "empty `loop {}`, which should block or sleep"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `while let` expressions on iterators.
    ///
    /// ### Why is this bad?
    /// Readability. A simple `for` loop is shorter and conveys
    /// the intent better.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for iterating a map (`HashMap` or `BTreeMap`) and
    /// ignoring either the keys or values.
    ///
    /// ### Why is this bad?
    /// Readability. There are `keys` and `values` methods that
    /// can be used to express that don't need the values or keys.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for loops that will always `break`, `return` or
    /// `continue` an outer loop.
    ///
    /// ### Why is this bad?
    /// This loop never loops, all it does is obfuscating the
    /// code.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for loops which have a range bound that is a mutable variable
    ///
    /// ### Why is this bad?
    /// One might think that modifying the mutable variable changes the loop bounds
    ///
    /// ### Known problems
    /// False positive when mutation is followed by a `break`, but the `break` is not immediately
    /// after the mutation:
    ///
    /// ```rust
    /// let mut x = 5;
    /// for _ in 0..x {
    ///     x += 1; // x is a range bound that is mutated
    ///     ..; // some other expression
    ///     break; // leaves the loop, so mutation is not an issue
    /// }
    /// ```
    ///
    /// False positive on nested loops ([#6072](https://github.com/rust-lang/rust-clippy/issues/6072))
    ///
    /// ### Example
    /// ```rust
    /// let mut foo = 42;
    /// for i in 0..foo {
    ///     foo -= 1;
    ///     println!("{}", i); // prints numbers from 0 to 42, not 0 to 21
    /// }
    /// ```
    pub MUT_RANGE_BOUND,
    suspicious,
    "for loop over a range where one of the bounds is a mutable variable"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks whether variables used within while loop condition
    /// can be (and are) mutated in the body.
    ///
    /// ### Why is this bad?
    /// If the condition is unchanged, entering the body of the loop
    /// will lead to an infinite loop.
    ///
    /// ### Known problems
    /// If the `while`-loop is in a closure, the check for mutation of the
    /// condition variables in the body can cause false negatives. For example when only `Upvar` `a` is
    /// in the condition and only `Upvar` `b` gets mutated in the body, the lint will not trigger.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks whether a for loop is being used to push a constant
    /// value into a Vec.
    ///
    /// ### Why is this bad?
    /// This kind of operation can be expressed more succinctly with
    /// `vec![item;SIZE]` or `vec.resize(NEW_SIZE, item)` and using these alternatives may also
    /// have better performance.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks whether a for loop has a single element.
    ///
    /// ### Why is this bad?
    /// There is no reason to have a loop of a
    /// single element.
    ///
    /// ### Example
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
    /// ### What it does
    /// Check for unnecessary `if let` usage in a for loop
    /// where only the `Some` or `Ok` variant of the iterator element is used.
    ///
    /// ### Why is this bad?
    /// It is verbose and can be simplified
    /// by first calling the `flatten` method on the `Iterator`.
    ///
    /// ### Example
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
        let for_loop = higher::ForLoop::hir(expr);
        if let Some(higher::ForLoop {
            pat,
            arg,
            body,
            loop_id,
            span,
        }) = for_loop
        {
            // we don't want to check expanded macros
            // this check is not at the top of the function
            // since higher::for_loop expressions are marked as expansions
            if body.span.from_expansion() {
                return;
            }
            check_for_loop(cx, pat, arg, body, expr, span);
            if let ExprKind::Block(block, _) = body.kind {
                never_loop::check(cx, block, loop_id, span, for_loop.as_ref());
            }
        }

        // we don't want to check expanded macros
        if expr.span.from_expansion() {
            return;
        }

        // check for never_loop
        if let ExprKind::Loop(block, ..) = expr.kind {
            never_loop::check(cx, block, expr.hir_id, expr.span, None);
        }

        // check for `loop { if let {} else break }` that could be `while let`
        // (also matches an explicit "match" instead of "if let")
        // (even if the "match" or "if let" is used for declaration)
        if let ExprKind::Loop(block, _, LoopSource::Loop, _) = expr.kind {
            // also check for empty `loop {}` statements, skipping those in #[panic_handler]
            empty_loop::check(cx, expr, block);
            while_let_loop::check(cx, expr, block);
        }

        while_let_on_iterator::check(cx, expr);

        if let Some(higher::While { condition, body }) = higher::While::hir(expr) {
            while_immutable_condition::check(cx, condition, body);
        }

        needless_collect::check(expr, cx);
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
    let is_manual_memcpy_triggered = manual_memcpy::check(cx, pat, arg, body, expr);
    if !is_manual_memcpy_triggered {
        needless_range_loop::check(cx, pat, arg, body, expr);
        explicit_counter_loop::check(cx, pat, arg, body, expr);
    }
    check_for_loop_arg(cx, pat, arg);
    for_kv_map::check(cx, pat, arg, body);
    mut_range_bound::check(cx, arg, body);
    single_element_loop::check(cx, pat, arg, body, expr);
    same_item_push::check(cx, pat, arg, body, expr);
    manual_flatten::check(cx, pat, arg, body, span);
}

fn check_for_loop_arg(cx: &LateContext<'_>, pat: &Pat<'_>, arg: &Expr<'_>) {
    let mut next_loop_linted = false; // whether or not ITER_NEXT_LOOP lint was used

    if let ExprKind::MethodCall(method, _, [self_arg], _) = arg.kind {
        let method_name = &*method.ident.as_str();
        // check for looping over x.iter() or x.iter_mut(), could use &x or &mut x
        match method_name {
            "iter" | "iter_mut" => explicit_iter_loop::check(cx, self_arg, arg, method_name),
            "into_iter" => {
                explicit_iter_loop::check(cx, self_arg, arg, method_name);
                explicit_into_iter_loop::check(cx, self_arg, arg);
            },
            "next" => {
                next_loop_linted = iter_next_loop::check(cx, arg);
            },
            _ => {},
        }
    }

    if !next_loop_linted {
        for_loops_over_fallibles::check(cx, pat, arg);
    }
}
