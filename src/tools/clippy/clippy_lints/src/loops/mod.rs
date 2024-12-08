mod empty_loop;
mod explicit_counter_loop;
mod explicit_into_iter_loop;
mod explicit_iter_loop;
mod for_kv_map;
mod infinite_loop;
mod iter_next_loop;
mod manual_find;
mod manual_flatten;
mod manual_memcpy;
mod manual_while_let_some;
mod missing_spin_loop;
mod mut_range_bound;
mod needless_range_loop;
mod never_loop;
mod same_item_push;
mod single_element_loop;
mod unused_enumerate_index;
mod utils;
mod while_float;
mod while_immutable_condition;
mod while_let_loop;
mod while_let_on_iterator;

use clippy_config::Conf;
use clippy_utils::higher;
use clippy_utils::msrvs::Msrv;
use rustc_ast::Label;
use rustc_hir::{Expr, ExprKind, LoopSource, Pat};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use utils::{IncrementVisitor, InitializeVisitor, make_iterator_snippet};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for for-loops that manually copy items between
    /// slices that could be optimized by having a memcpy.
    ///
    /// ### Why is this bad?
    /// It is not as fast as a memcpy.
    ///
    /// ### Example
    /// ```no_run
    /// # let src = vec![1];
    /// # let mut dst = vec![0; 65];
    /// for i in 0..src.len() {
    ///     dst[i + 64] = src[i];
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let src = vec![1];
    /// # let mut dst = vec![0; 65];
    /// dst[64..(src.len() + 64)].clone_from_slice(&src[..]);
    /// ```
    #[clippy::version = "pre 1.29.0"]
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
    /// more clear and is probably faster because it eliminates
    /// the bounds check that is done when indexing.
    ///
    /// ### Example
    /// ```no_run
    /// let vec = vec!['a', 'b', 'c'];
    /// for i in 0..vec.len() {
    ///     println!("{}", vec[i]);
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let vec = vec!['a', 'b', 'c'];
    /// for i in vec {
    ///     println!("{}", i);
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
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
    /// ```no_run
    /// // with `y` a `Vec` or slice:
    /// # let y = vec![1];
    /// for x in y.iter() {
    ///     // ..
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let y = vec![1];
    /// for x in &y {
    ///     // ..
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
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
    /// ```no_run
    /// # let y = vec![1];
    /// // with `y` a `Vec` or slice:
    /// for x in y.into_iter() {
    ///     // ..
    /// }
    /// ```
    /// can be rewritten to
    /// ```no_run
    /// # let y = vec![1];
    /// for x in y {
    ///     // ..
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
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
    #[clippy::version = "pre 1.29.0"]
    pub ITER_NEXT_LOOP,
    correctness,
    "for-looping over `_.next()` which is probably not intended"
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
    /// ### Example
    /// ```rust,no_run
    /// let y = Some(1);
    /// loop {
    ///     let x = match y {
    ///         Some(x) => x,
    ///         None => break,
    ///     };
    ///     // ..
    /// }
    /// ```
    /// Use instead:
    /// ```rust,no_run
    /// let y = Some(1);
    /// while let Some(x) = y {
    ///     // ..
    /// };
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub WHILE_LET_LOOP,
    complexity,
    "`loop { if let { ... } else break }`, which can be written as a `while let` loop"
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
    /// ```no_run
    /// # let v = vec![1];
    /// # fn bar(bar: usize, baz: usize) {}
    /// let mut i = 0;
    /// for item in &v {
    ///     bar(i, *item);
    ///     i += 1;
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let v = vec![1];
    /// # fn bar(bar: usize, baz: usize) {}
    /// for (i, item) in v.iter().enumerate() { bar(i, *item); }
    /// ```
    #[clippy::version = "pre 1.29.0"]
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
    #[clippy::version = "pre 1.29.0"]
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
    /// while let Some(val) = iter.next() {
    ///     ..
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```ignore
    /// for val in &mut iter {
    ///     ..
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
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
    #[clippy::version = "pre 1.29.0"]
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
    /// ```no_run
    /// loop {
    ///     ..;
    ///     break;
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEVER_LOOP,
    correctness,
    "any loop that will always `break` or `return`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for loops with a range bound that is a mutable variable.
    ///
    /// ### Why is this bad?
    /// One might think that modifying the mutable variable changes the loop bounds. It doesn't.
    ///
    /// ### Known problems
    /// False positive when mutation is followed by a `break`, but the `break` is not immediately
    /// after the mutation:
    ///
    /// ```no_run
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
    /// ```no_run
    /// let mut foo = 42;
    /// for i in 0..foo {
    ///     foo -= 1;
    ///     println!("{i}"); // prints numbers from 0 to 41, not 0 to 21
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
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
    /// ```no_run
    /// let i = 0;
    /// while i > 10 {
    ///     println!("let me loop forever!");
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub WHILE_IMMUTABLE_CONDITION,
    correctness,
    "variables used within while expression are not mutated in the body"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for while loops comparing floating point values.
    ///
    /// ### Why is this bad?
    /// If you increment floating point values, errors can compound,
    /// so, use integers instead if possible.
    ///
    /// ### Known problems
    /// The lint will catch all while loops comparing floating point
    /// values without regarding the increment.
    ///
    /// ### Example
    /// ```no_run
    /// let mut x = 0.0;
    /// while x < 42.0 {
    ///     x += 1.0;
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let mut x = 0;
    /// while x < 42 {
    ///     x += 1;
    /// }
    /// ```
    #[clippy::version = "1.80.0"]
    pub WHILE_FLOAT,
    nursery,
    "while loops comparing floating point values"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks whether a for loop is being used to push a constant
    /// value into a Vec.
    ///
    /// ### Why is this bad?
    /// This kind of operation can be expressed more succinctly with
    /// `vec![item; SIZE]` or `vec.resize(NEW_SIZE, item)` and using these alternatives may also
    /// have better performance.
    ///
    /// ### Example
    /// ```no_run
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
    ///
    /// Use instead:
    /// ```no_run
    /// let item1 = 2;
    /// let item2 = 3;
    /// let mut vec: Vec<u8> = vec![item1; 20];
    /// vec.resize(20 + 30, item2);
    /// ```
    #[clippy::version = "1.47.0"]
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
    /// ```no_run
    /// let item1 = 2;
    /// for item in &[item1] {
    ///     println!("{}", item);
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let item1 = 2;
    /// let item = &item1;
    /// println!("{}", item);
    /// ```
    #[clippy::version = "1.49.0"]
    pub SINGLE_ELEMENT_LOOP,
    complexity,
    "there is no reason to have a single element loop"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unnecessary `if let` usage in a for loop
    /// where only the `Some` or `Ok` variant of the iterator element is used.
    ///
    /// ### Why is this bad?
    /// It is verbose and can be simplified
    /// by first calling the `flatten` method on the `Iterator`.
    ///
    /// ### Example
    ///
    /// ```no_run
    /// let x = vec![Some(1), Some(2), Some(3)];
    /// for n in x {
    ///     if let Some(n) = n {
    ///         println!("{}", n);
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x = vec![Some(1), Some(2), Some(3)];
    /// for n in x.into_iter().flatten() {
    ///     println!("{}", n);
    /// }
    /// ```
    #[clippy::version = "1.52.0"]
    pub MANUAL_FLATTEN,
    complexity,
    "for loops over `Option`s or `Result`s with a single expression can be simplified"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for empty spin loops
    ///
    /// ### Why is this bad?
    /// The loop body should have something like `thread::park()` or at least
    /// `std::hint::spin_loop()` to avoid needlessly burning cycles and conserve
    /// energy. Perhaps even better use an actual lock, if possible.
    ///
    /// ### Known problems
    /// This lint doesn't currently trigger on `while let` or
    /// `loop { match .. { .. } }` loops, which would be considered idiomatic in
    /// combination with e.g. `AtomicBool::compare_exchange_weak`.
    ///
    /// ### Example
    ///
    /// ```ignore
    /// use core::sync::atomic::{AtomicBool, Ordering};
    /// let b = AtomicBool::new(true);
    /// // give a ref to `b` to another thread,wait for it to become false
    /// while b.load(Ordering::Acquire) {};
    /// ```
    /// Use instead:
    /// ```rust,no_run
    ///# use core::sync::atomic::{AtomicBool, Ordering};
    ///# let b = AtomicBool::new(true);
    /// while b.load(Ordering::Acquire) {
    ///     std::hint::spin_loop()
    /// }
    /// ```
    #[clippy::version = "1.61.0"]
    pub MISSING_SPIN_LOOP,
    perf,
    "An empty busy waiting loop"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual implementations of Iterator::find
    ///
    /// ### Why is this bad?
    /// It doesn't affect performance, but using `find` is shorter and easier to read.
    ///
    /// ### Example
    ///
    /// ```no_run
    /// fn example(arr: Vec<i32>) -> Option<i32> {
    ///     for el in arr {
    ///         if el == 1 {
    ///             return Some(el);
    ///         }
    ///     }
    ///     None
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn example(arr: Vec<i32>) -> Option<i32> {
    ///     arr.into_iter().find(|&el| el == 1)
    /// }
    /// ```
    #[clippy::version = "1.64.0"]
    pub MANUAL_FIND,
    complexity,
    "manual implementation of `Iterator::find`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for uses of the `enumerate` method where the index is unused (`_`)
    ///
    /// ### Why is this bad?
    /// The index from `.enumerate()` is immediately dropped.
    ///
    /// ### Example
    /// ```rust
    /// let v = vec![1, 2, 3, 4];
    /// for (_, x) in v.iter().enumerate() {
    ///     println!("{x}");
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// let v = vec![1, 2, 3, 4];
    /// for x in v.iter() {
    ///     println!("{x}");
    /// }
    /// ```
    #[clippy::version = "1.75.0"]
    pub UNUSED_ENUMERATE_INDEX,
    style,
    "using `.enumerate()` and immediately dropping the index"
}

declare_clippy_lint! {
    /// ### What it does
    /// Looks for loops that check for emptiness of a `Vec` in the condition and pop an element
    /// in the body as a separate operation.
    ///
    /// ### Why is this bad?
    /// Such loops can be written in a more idiomatic way by using a while-let loop and directly
    /// pattern matching on the return value of `Vec::pop()`.
    ///
    /// ### Example
    /// ```no_run
    /// let mut numbers = vec![1, 2, 3, 4, 5];
    /// while !numbers.is_empty() {
    ///     let number = numbers.pop().unwrap();
    ///     // use `number`
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// let mut numbers = vec![1, 2, 3, 4, 5];
    /// while let Some(number) = numbers.pop() {
    ///     // use `number`
    /// }
    /// ```
    #[clippy::version = "1.71.0"]
    pub MANUAL_WHILE_LET_SOME,
    style,
    "checking for emptiness of a `Vec` in the loop condition and popping an element in the body"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for infinite loops in a function where the return type is not `!`
    /// and lint accordingly.
    ///
    /// ### Why restrict this?
    /// Making the return type `!` serves as documentation that the function does not return.
    /// If the function is not intended to loop infinitely, then this lint may detect a bug.
    ///
    /// ### Example
    /// ```no_run,ignore
    /// fn run_forever() {
    ///     loop {
    ///         // do something
    ///     }
    /// }
    /// ```
    /// If infinite loops are as intended:
    /// ```no_run,ignore
    /// fn run_forever() -> ! {
    ///     loop {
    ///         // do something
    ///     }
    /// }
    /// ```
    /// Otherwise add a `break` or `return` condition:
    /// ```no_run,ignore
    /// fn run_forever() {
    ///     loop {
    ///         // do something
    ///         if condition {
    ///             break;
    ///         }
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.76.0"]
    pub INFINITE_LOOP,
    restriction,
    "possibly unintended infinite loop"
}

pub struct Loops {
    msrv: Msrv,
    enforce_iter_loop_reborrow: bool,
}
impl Loops {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: conf.msrv.clone(),
            enforce_iter_loop_reborrow: conf.enforce_iter_loop_reborrow,
        }
    }
}

impl_lint_pass!(Loops => [
    MANUAL_MEMCPY,
    MANUAL_FLATTEN,
    NEEDLESS_RANGE_LOOP,
    EXPLICIT_ITER_LOOP,
    EXPLICIT_INTO_ITER_LOOP,
    ITER_NEXT_LOOP,
    WHILE_LET_LOOP,
    EXPLICIT_COUNTER_LOOP,
    EMPTY_LOOP,
    WHILE_LET_ON_ITERATOR,
    FOR_KV_MAP,
    NEVER_LOOP,
    MUT_RANGE_BOUND,
    WHILE_IMMUTABLE_CONDITION,
    WHILE_FLOAT,
    SAME_ITEM_PUSH,
    SINGLE_ELEMENT_LOOP,
    MISSING_SPIN_LOOP,
    MANUAL_FIND,
    MANUAL_WHILE_LET_SOME,
    UNUSED_ENUMERATE_INDEX,
    INFINITE_LOOP,
]);

impl<'tcx> LateLintPass<'tcx> for Loops {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let for_loop = higher::ForLoop::hir(expr);
        if let Some(higher::ForLoop {
            pat,
            arg,
            body,
            loop_id,
            span,
            label,
        }) = for_loop
        {
            // we don't want to check expanded macros
            // this check is not at the top of the function
            // since higher::for_loop expressions are marked as expansions
            if body.span.from_expansion() {
                return;
            }
            self.check_for_loop(cx, pat, arg, body, expr, span, label);
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
        if let ExprKind::Loop(block, label, LoopSource::Loop, _) = expr.kind {
            // also check for empty `loop {}` statements, skipping those in #[panic_handler]
            empty_loop::check(cx, expr, block);
            while_let_loop::check(cx, expr, block);
            infinite_loop::check(cx, expr, block, label);
        }

        while_let_on_iterator::check(cx, expr);

        if let Some(higher::While { condition, body, span }) = higher::While::hir(expr) {
            while_immutable_condition::check(cx, condition, body);
            while_float::check(cx, condition);
            missing_spin_loop::check(cx, condition, body);
            manual_while_let_some::check(cx, condition, body, span);
        }
    }

    extract_msrv_attr!(LateContext);
}

impl Loops {
    #[allow(clippy::too_many_arguments)]
    fn check_for_loop<'tcx>(
        &self,
        cx: &LateContext<'tcx>,
        pat: &'tcx Pat<'_>,
        arg: &'tcx Expr<'_>,
        body: &'tcx Expr<'_>,
        expr: &'tcx Expr<'_>,
        span: Span,
        label: Option<Label>,
    ) {
        let is_manual_memcpy_triggered = manual_memcpy::check(cx, pat, arg, body, expr);
        if !is_manual_memcpy_triggered {
            needless_range_loop::check(cx, pat, arg, body, expr);
            explicit_counter_loop::check(cx, pat, arg, body, expr, label);
        }
        self.check_for_loop_arg(cx, pat, arg);
        for_kv_map::check(cx, pat, arg, body);
        mut_range_bound::check(cx, arg, body);
        single_element_loop::check(cx, pat, arg, body, expr);
        same_item_push::check(cx, pat, arg, body, expr);
        manual_flatten::check(cx, pat, arg, body, span);
        manual_find::check(cx, pat, arg, body, span, expr);
        unused_enumerate_index::check(cx, pat, arg, body);
    }

    fn check_for_loop_arg(&self, cx: &LateContext<'_>, _: &Pat<'_>, arg: &Expr<'_>) {
        if let ExprKind::MethodCall(method, self_arg, [], _) = arg.kind {
            match method.ident.as_str() {
                "iter" | "iter_mut" => {
                    explicit_iter_loop::check(cx, self_arg, arg, &self.msrv, self.enforce_iter_loop_reborrow);
                },
                "into_iter" => {
                    explicit_into_iter_loop::check(cx, self_arg, arg);
                },
                "next" => {
                    iter_next_loop::check(cx, arg);
                },
                _ => {},
            }
        }
    }
}
