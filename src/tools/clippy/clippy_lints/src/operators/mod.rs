mod absurd_extreme_comparisons;
mod assign_op_pattern;
mod bit_mask;
mod cmp_nan;
mod cmp_owned;
mod double_comparison;
mod duration_subsec;
mod eq_op;
mod erasing_op;
mod float_cmp;
mod float_equality_without_abs;
mod identity_op;
mod integer_division;
mod misrefactored_assign_op;
mod modulo_arithmetic;
mod modulo_one;
mod needless_bitwise_bool;
mod numeric_arithmetic;
mod op_ref;
mod ptr_eq;
mod self_assignment;
mod verbose_bit_mask;

pub(crate) mod arithmetic_side_effects;

use rustc_hir::{Body, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for comparisons where one side of the relation is
    /// either the minimum or maximum value for its type and warns if it involves a
    /// case that is always true or always false. Only integer and boolean types are
    /// checked.
    ///
    /// ### Why is this bad?
    /// An expression like `min <= x` may misleadingly imply
    /// that it is possible for `x` to be less than the minimum. Expressions like
    /// `max < x` are probably mistakes.
    ///
    /// ### Known problems
    /// For `usize` the size of the current compile target will
    /// be assumed (e.g., 64 bits on 64 bit systems). This means code that uses such
    /// a comparison to detect target pointer width will trigger this lint. One can
    /// use `mem::sizeof` and compare its value or conditional compilation
    /// attributes
    /// like `#[cfg(target_pointer_width = "64")] ..` instead.
    ///
    /// ### Example
    /// ```rust
    /// let vec: Vec<isize> = Vec::new();
    /// if vec.len() <= 0 {}
    /// if 100 > i32::MAX {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ABSURD_EXTREME_COMPARISONS,
    correctness,
    "a comparison with a maximum or minimum value that is always true or false"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks any kind of arithmetic operation of any type.
    ///
    /// Operators like `+`, `-`, `*` or `<<` are usually capable of overflowing according to the [Rust
    /// Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#overflow),
    /// or can panic (`/`, `%`).
    ///
    /// Known safe built-in types like `Wrapping` or `Saturating`, floats, operations in constant
    /// environments, allowed types and non-constant operations that won't overflow are ignored.
    ///
    /// ### Why is this bad?
    /// For integers, overflow will trigger a panic in debug builds or wrap the result in
    /// release mode; division by zero will cause a panic in either mode. As a result, it is
    /// desirable to explicitly call checked, wrapping or saturating arithmetic methods.
    ///
    /// #### Example
    /// ```rust
    /// // `n` can be any number, including `i32::MAX`.
    /// fn foo(n: i32) -> i32 {
    ///   n + 1
    /// }
    /// ```
    ///
    /// Third-party types can also overflow or present unwanted side-effects.
    ///
    /// #### Example
    /// ```ignore,rust
    /// use rust_decimal::Decimal;
    /// let _n = Decimal::MAX + Decimal::MAX;
    /// ```
    #[clippy::version = "1.64.0"]
    pub ARITHMETIC_SIDE_EFFECTS,
    restriction,
    "any arithmetic expression that can cause side effects like overflows or panics"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for integer arithmetic operations which could overflow or panic.
    ///
    /// Specifically, checks for any operators (`+`, `-`, `*`, `<<`, etc) which are capable
    /// of overflowing according to the [Rust
    /// Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#overflow),
    /// or which can panic (`/`, `%`). No bounds analysis or sophisticated reasoning is
    /// attempted.
    ///
    /// ### Why is this bad?
    /// Integer overflow will trigger a panic in debug builds or will wrap in
    /// release mode. Division by zero will cause a panic in either mode. In some applications one
    /// wants explicitly checked, wrapping or saturating arithmetic.
    ///
    /// ### Example
    /// ```rust
    /// # let a = 0;
    /// a + 1;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INTEGER_ARITHMETIC,
    restriction,
    "any integer arithmetic expression which could overflow or panic"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for float arithmetic.
    ///
    /// ### Why is this bad?
    /// For some embedded systems or kernel development, it
    /// can be useful to rule out floating-point numbers.
    ///
    /// ### Example
    /// ```rust
    /// # let a = 0.0;
    /// a + 1.0;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub FLOAT_ARITHMETIC,
    restriction,
    "any floating-point arithmetic statement"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `a = a op b` or `a = b commutative_op a`
    /// patterns.
    ///
    /// ### Why is this bad?
    /// These can be written as the shorter `a op= b`.
    ///
    /// ### Known problems
    /// While forbidden by the spec, `OpAssign` traits may have
    /// implementations that differ from the regular `Op` impl.
    ///
    /// ### Example
    /// ```rust
    /// let mut a = 5;
    /// let b = 0;
    /// // ...
    ///
    /// a = a + b;
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let mut a = 5;
    /// let b = 0;
    /// // ...
    ///
    /// a += b;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ASSIGN_OP_PATTERN,
    style,
    "assigning the result of an operation on a variable to that same variable"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `a op= a op b` or `a op= b op a` patterns.
    ///
    /// ### Why is this bad?
    /// Most likely these are bugs where one meant to write `a
    /// op= b`.
    ///
    /// ### Known problems
    /// Clippy cannot know for sure if `a op= a op b` should have
    /// been `a = a op a op b` or `a = a op b`/`a op= b`. Therefore, it suggests both.
    /// If `a op= a op b` is really the correct behavior it should be
    /// written as `a = a op a op b` as it's less confusing.
    ///
    /// ### Example
    /// ```rust
    /// let mut a = 5;
    /// let b = 2;
    /// // ...
    /// a += a + b;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MISREFACTORED_ASSIGN_OP,
    suspicious,
    "having a variable on both sides of an assign op"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for incompatible bit masks in comparisons.
    ///
    /// The formula for detecting if an expression of the type `_ <bit_op> m
    /// <cmp_op> c` (where `<bit_op>` is one of {`&`, `|`} and `<cmp_op>` is one of
    /// {`!=`, `>=`, `>`, `!=`, `>=`, `>`}) can be determined from the following
    /// table:
    ///
    /// |Comparison  |Bit Op|Example      |is always|Formula               |
    /// |------------|------|-------------|---------|----------------------|
    /// |`==` or `!=`| `&`  |`x & 2 == 3` |`false`  |`c & m != c`          |
    /// |`<`  or `>=`| `&`  |`x & 2 < 3`  |`true`   |`m < c`               |
    /// |`>`  or `<=`| `&`  |`x & 1 > 1`  |`false`  |`m <= c`              |
    /// |`==` or `!=`| `\|` |`x \| 1 == 0`|`false`  |`c \| m != c`         |
    /// |`<`  or `>=`| `\|` |`x \| 1 < 1` |`false`  |`m >= c`              |
    /// |`<=` or `>` | `\|` |`x \| 1 > 0` |`true`   |`m > c`               |
    ///
    /// ### Why is this bad?
    /// If the bits that the comparison cares about are always
    /// set to zero or one by the bit mask, the comparison is constant `true` or
    /// `false` (depending on mask, compared value, and operators).
    ///
    /// So the code is actively misleading, and the only reason someone would write
    /// this intentionally is to win an underhanded Rust contest or create a
    /// test-case for this lint.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// if (x & 1 == 2) { }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub BAD_BIT_MASK,
    correctness,
    "expressions of the form `_ & mask == select` that will only ever return `true` or `false`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for bit masks in comparisons which can be removed
    /// without changing the outcome. The basic structure can be seen in the
    /// following table:
    ///
    /// |Comparison| Bit Op   |Example     |equals |
    /// |----------|----------|------------|-------|
    /// |`>` / `<=`|`\|` / `^`|`x \| 2 > 3`|`x > 3`|
    /// |`<` / `>=`|`\|` / `^`|`x ^ 1 < 4` |`x < 4`|
    ///
    /// ### Why is this bad?
    /// Not equally evil as [`bad_bit_mask`](#bad_bit_mask),
    /// but still a bit misleading, because the bit mask is ineffective.
    ///
    /// ### Known problems
    /// False negatives: This lint will only match instances
    /// where we have figured out the math (which is for a power-of-two compared
    /// value). This means things like `x | 1 >= 7` (which would be better written
    /// as `x >= 6`) will not be reported (but bit masks like this are fairly
    /// uncommon).
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// if (x | 1 > 3) {  }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INEFFECTIVE_BIT_MASK,
    correctness,
    "expressions where a bit mask will be rendered useless by a comparison, e.g., `(x | 1) > 2`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for bit masks that can be replaced by a call
    /// to `trailing_zeros`
    ///
    /// ### Why is this bad?
    /// `x.trailing_zeros() > 4` is much clearer than `x & 15
    /// == 0`
    ///
    /// ### Known problems
    /// llvm generates better code for `x & 15 == 0` on x86
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// if x & 0b1111 == 0 { }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub VERBOSE_BIT_MASK,
    pedantic,
    "expressions where a bit mask is less readable than the corresponding method call"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for double comparisons that could be simplified to a single expression.
    ///
    ///
    /// ### Why is this bad?
    /// Readability.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// # let y = 2;
    /// if x == y || x < y {}
    /// ```
    ///
    /// Use instead:
    ///
    /// ```rust
    /// # let x = 1;
    /// # let y = 2;
    /// if x <= y {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DOUBLE_COMPARISONS,
    complexity,
    "unnecessary double comparisons that can be simplified"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calculation of subsecond microseconds or milliseconds
    /// from other `Duration` methods.
    ///
    /// ### Why is this bad?
    /// It's more concise to call `Duration::subsec_micros()` or
    /// `Duration::subsec_millis()` than to calculate them.
    ///
    /// ### Example
    /// ```rust
    /// # use std::time::Duration;
    /// # let duration = Duration::new(5, 0);
    /// let micros = duration.subsec_nanos() / 1_000;
    /// let millis = duration.subsec_nanos() / 1_000_000;
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # use std::time::Duration;
    /// # let duration = Duration::new(5, 0);
    /// let micros = duration.subsec_micros();
    /// let millis = duration.subsec_millis();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DURATION_SUBSEC,
    complexity,
    "checks for calculation of subsecond microseconds or milliseconds"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for equal operands to comparison, logical and
    /// bitwise, difference and division binary operators (`==`, `>`, etc., `&&`,
    /// `||`, `&`, `|`, `^`, `-` and `/`).
    ///
    /// ### Why is this bad?
    /// This is usually just a typo or a copy and paste error.
    ///
    /// ### Known problems
    /// False negatives: We had some false positives regarding
    /// calls (notably [racer](https://github.com/phildawes/racer) had one instance
    /// of `x.pop() && x.pop()`), so we removed matching any function or method
    /// calls. We may introduce a list of known pure functions in the future.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// if x + 1 == x + 1 {}
    ///
    /// // or
    ///
    /// # let a = 3;
    /// # let b = 4;
    /// assert_eq!(a, a);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub EQ_OP,
    correctness,
    "equal operands on both sides of a comparison or bitwise combination (e.g., `x == x`)"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for arguments to `==` which have their address
    /// taken to satisfy a bound
    /// and suggests to dereference the other argument instead
    ///
    /// ### Why is this bad?
    /// It is more idiomatic to dereference the other argument.
    ///
    /// ### Example
    /// ```rust,ignore
    /// &x == y
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// x == *y
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub OP_REF,
    style,
    "taking a reference to satisfy the type constraints on `==`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for erasing operations, e.g., `x * 0`.
    ///
    /// ### Why is this bad?
    /// The whole expression can be replaced by zero.
    /// This is most likely not the intended outcome and should probably be
    /// corrected
    ///
    /// ### Example
    /// ```rust
    /// let x = 1;
    /// 0 / x;
    /// 0 * x;
    /// x & 0;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ERASING_OP,
    correctness,
    "using erasing operations, e.g., `x * 0` or `y & 0`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for statements of the form `(a - b) < f32::EPSILON` or
    /// `(a - b) < f64::EPSILON`. Notes the missing `.abs()`.
    ///
    /// ### Why is this bad?
    /// The code without `.abs()` is more likely to have a bug.
    ///
    /// ### Known problems
    /// If the user can ensure that b is larger than a, the `.abs()` is
    /// technically unnecessary. However, it will make the code more robust and doesn't have any
    /// large performance implications. If the abs call was deliberately left out for performance
    /// reasons, it is probably better to state this explicitly in the code, which then can be done
    /// with an allow.
    ///
    /// ### Example
    /// ```rust
    /// pub fn is_roughly_equal(a: f32, b: f32) -> bool {
    ///     (a - b) < f32::EPSILON
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// pub fn is_roughly_equal(a: f32, b: f32) -> bool {
    ///     (a - b).abs() < f32::EPSILON
    /// }
    /// ```
    #[clippy::version = "1.48.0"]
    pub FLOAT_EQUALITY_WITHOUT_ABS,
    suspicious,
    "float equality check without `.abs()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for identity operations, e.g., `x + 0`.
    ///
    /// ### Why is this bad?
    /// This code can be removed without changing the
    /// meaning. So it just obscures what's going on. Delete it mercilessly.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// x / 1 + 0 * 1 - 0 | 0;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub IDENTITY_OP,
    complexity,
    "using identity operations, e.g., `x + 0` or `y / 1`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for division of integers
    ///
    /// ### Why is this bad?
    /// When outside of some very specific algorithms,
    /// integer division is very often a mistake because it discards the
    /// remainder.
    ///
    /// ### Example
    /// ```rust
    /// let x = 3 / 2;
    /// println!("{}", x);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let x = 3f32 / 2f32;
    /// println!("{}", x);
    /// ```
    #[clippy::version = "1.37.0"]
    pub INTEGER_DIVISION,
    restriction,
    "integer division may cause loss of precision"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for comparisons to NaN.
    ///
    /// ### Why is this bad?
    /// NaN does not compare meaningfully to anything – not
    /// even itself – so those comparisons are simply wrong.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1.0;
    /// if x == f32::NAN { }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let x = 1.0f32;
    /// if x.is_nan() { }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CMP_NAN,
    correctness,
    "comparisons to `NAN`, which will always return false, probably not intended"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for conversions to owned values just for the sake
    /// of a comparison.
    ///
    /// ### Why is this bad?
    /// The comparison can operate on a reference, so creating
    /// an owned value effectively throws it away directly afterwards, which is
    /// needlessly consuming code and heap space.
    ///
    /// ### Example
    /// ```rust
    /// # let x = "foo";
    /// # let y = String::from("foo");
    /// if x.to_owned() == y {}
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let x = "foo";
    /// # let y = String::from("foo");
    /// if x == y {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CMP_OWNED,
    perf,
    "creating owned instances for comparing with others, e.g., `x == \"foo\".to_string()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for (in-)equality comparisons on floating-point
    /// values (apart from zero), except in functions called `*eq*` (which probably
    /// implement equality for a type involving floats).
    ///
    /// ### Why is this bad?
    /// Floating point calculations are usually imprecise, so
    /// asking if two values are *exactly* equal is asking for trouble. For a good
    /// guide on what to do, see [the floating point
    /// guide](http://www.floating-point-gui.de/errors/comparison).
    ///
    /// ### Example
    /// ```rust
    /// let x = 1.2331f64;
    /// let y = 1.2332f64;
    ///
    /// if y == 1.23f64 { }
    /// if y != x {} // where both are floats
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let x = 1.2331f64;
    /// # let y = 1.2332f64;
    /// let error_margin = f64::EPSILON; // Use an epsilon for comparison
    /// // Or, if Rust <= 1.42, use `std::f64::EPSILON` constant instead.
    /// // let error_margin = std::f64::EPSILON;
    /// if (y - 1.23f64).abs() < error_margin { }
    /// if (y - x).abs() > error_margin { }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub FLOAT_CMP,
    pedantic,
    "using `==` or `!=` on float values instead of comparing difference with an epsilon"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for (in-)equality comparisons on floating-point
    /// value and constant, except in functions called `*eq*` (which probably
    /// implement equality for a type involving floats).
    ///
    /// ### Why is this bad?
    /// Floating point calculations are usually imprecise, so
    /// asking if two values are *exactly* equal is asking for trouble. For a good
    /// guide on what to do, see [the floating point
    /// guide](http://www.floating-point-gui.de/errors/comparison).
    ///
    /// ### Example
    /// ```rust
    /// let x: f64 = 1.0;
    /// const ONE: f64 = 1.00;
    ///
    /// if x == ONE { } // where both are floats
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let x: f64 = 1.0;
    /// # const ONE: f64 = 1.00;
    /// let error_margin = f64::EPSILON; // Use an epsilon for comparison
    /// // Or, if Rust <= 1.42, use `std::f64::EPSILON` constant instead.
    /// // let error_margin = std::f64::EPSILON;
    /// if (x - ONE).abs() < error_margin { }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub FLOAT_CMP_CONST,
    restriction,
    "using `==` or `!=` on float constants instead of comparing difference with an epsilon"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for getting the remainder of a division by one or minus
    /// one.
    ///
    /// ### Why is this bad?
    /// The result for a divisor of one can only ever be zero; for
    /// minus one it can cause panic/overflow (if the left operand is the minimal value of
    /// the respective integer type) or results in zero. No one will write such code
    /// deliberately, unless trying to win an Underhanded Rust Contest. Even for that
    /// contest, it's probably a bad idea. Use something more underhanded.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// let a = x % 1;
    /// let a = x % -1;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MODULO_ONE,
    correctness,
    "taking a number modulo +/-1, which can either panic/overflow or always returns 0"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for modulo arithmetic.
    ///
    /// ### Why is this bad?
    /// The results of modulo (%) operation might differ
    /// depending on the language, when negative numbers are involved.
    /// If you interop with different languages it might be beneficial
    /// to double check all places that use modulo arithmetic.
    ///
    /// For example, in Rust `17 % -3 = 2`, but in Python `17 % -3 = -1`.
    ///
    /// ### Example
    /// ```rust
    /// let x = -17 % 3;
    /// ```
    #[clippy::version = "1.42.0"]
    pub MODULO_ARITHMETIC,
    restriction,
    "any modulo arithmetic statement"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for uses of bitwise and/or operators between booleans, where performance may be improved by using
    /// a lazy and.
    ///
    /// ### Why is this bad?
    /// The bitwise operators do not support short-circuiting, so it may hinder code performance.
    /// Additionally, boolean logic "masked" as bitwise logic is not caught by lints like `unnecessary_fold`
    ///
    /// ### Known problems
    /// This lint evaluates only when the right side is determined to have no side effects. At this time, that
    /// determination is quite conservative.
    ///
    /// ### Example
    /// ```rust
    /// let (x,y) = (true, false);
    /// if x & !y {} // where both x and y are booleans
    /// ```
    /// Use instead:
    /// ```rust
    /// let (x,y) = (true, false);
    /// if x && !y {}
    /// ```
    #[clippy::version = "1.54.0"]
    pub NEEDLESS_BITWISE_BOOL,
    pedantic,
    "Boolean expressions that use bitwise rather than lazy operators"
}

declare_clippy_lint! {
    /// ### What it does
    /// Use `std::ptr::eq` when applicable
    ///
    /// ### Why is this bad?
    /// `ptr::eq` can be used to compare `&T` references
    /// (which coerce to `*const T` implicitly) by their address rather than
    /// comparing the values they point to.
    ///
    /// ### Example
    /// ```rust
    /// let a = &[1, 2, 3];
    /// let b = &[1, 2, 3];
    ///
    /// assert!(a as *const _ as usize == b as *const _ as usize);
    /// ```
    /// Use instead:
    /// ```rust
    /// let a = &[1, 2, 3];
    /// let b = &[1, 2, 3];
    ///
    /// assert!(std::ptr::eq(a, b));
    /// ```
    #[clippy::version = "1.49.0"]
    pub PTR_EQ,
    style,
    "use `std::ptr::eq` when comparing raw pointers"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for explicit self-assignments.
    ///
    /// ### Why is this bad?
    /// Self-assignments are redundant and unlikely to be
    /// intentional.
    ///
    /// ### Known problems
    /// If expression contains any deref coercions or
    /// indexing operations they are assumed not to have any side effects.
    ///
    /// ### Example
    /// ```rust
    /// struct Event {
    ///     x: i32,
    /// }
    ///
    /// fn copy_position(a: &mut Event, b: &Event) {
    ///     a.x = a.x;
    /// }
    /// ```
    ///
    /// Should be:
    /// ```rust
    /// struct Event {
    ///     x: i32,
    /// }
    ///
    /// fn copy_position(a: &mut Event, b: &Event) {
    ///     a.x = b.x;
    /// }
    /// ```
    #[clippy::version = "1.48.0"]
    pub SELF_ASSIGNMENT,
    correctness,
    "explicit self-assignment"
}

pub struct Operators {
    arithmetic_context: numeric_arithmetic::Context,
    verbose_bit_mask_threshold: u64,
}
impl_lint_pass!(Operators => [
    ABSURD_EXTREME_COMPARISONS,
    ARITHMETIC_SIDE_EFFECTS,
    INTEGER_ARITHMETIC,
    FLOAT_ARITHMETIC,
    ASSIGN_OP_PATTERN,
    MISREFACTORED_ASSIGN_OP,
    BAD_BIT_MASK,
    INEFFECTIVE_BIT_MASK,
    VERBOSE_BIT_MASK,
    DOUBLE_COMPARISONS,
    DURATION_SUBSEC,
    EQ_OP,
    OP_REF,
    ERASING_OP,
    FLOAT_EQUALITY_WITHOUT_ABS,
    IDENTITY_OP,
    INTEGER_DIVISION,
    CMP_NAN,
    CMP_OWNED,
    FLOAT_CMP,
    FLOAT_CMP_CONST,
    MODULO_ONE,
    MODULO_ARITHMETIC,
    NEEDLESS_BITWISE_BOOL,
    PTR_EQ,
    SELF_ASSIGNMENT,
]);
impl Operators {
    pub fn new(verbose_bit_mask_threshold: u64) -> Self {
        Self {
            arithmetic_context: numeric_arithmetic::Context::default(),
            verbose_bit_mask_threshold,
        }
    }
}
impl<'tcx> LateLintPass<'tcx> for Operators {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        eq_op::check_assert(cx, e);
        match e.kind {
            ExprKind::Binary(op, lhs, rhs) => {
                if !e.span.from_expansion() {
                    absurd_extreme_comparisons::check(cx, e, op.node, lhs, rhs);
                    if !(macro_with_not_op(lhs) || macro_with_not_op(rhs)) {
                        eq_op::check(cx, e, op.node, lhs, rhs);
                        op_ref::check(cx, e, op.node, lhs, rhs);
                    }
                    erasing_op::check(cx, e, op.node, lhs, rhs);
                    identity_op::check(cx, e, op.node, lhs, rhs);
                    needless_bitwise_bool::check(cx, e, op.node, lhs, rhs);
                    ptr_eq::check(cx, e, op.node, lhs, rhs);
                }
                self.arithmetic_context.check_binary(cx, e, op.node, lhs, rhs);
                bit_mask::check(cx, e, op.node, lhs, rhs);
                verbose_bit_mask::check(cx, e, op.node, lhs, rhs, self.verbose_bit_mask_threshold);
                double_comparison::check(cx, op.node, lhs, rhs, e.span);
                duration_subsec::check(cx, e, op.node, lhs, rhs);
                float_equality_without_abs::check(cx, e, op.node, lhs, rhs);
                integer_division::check(cx, e, op.node, lhs, rhs);
                cmp_nan::check(cx, e, op.node, lhs, rhs);
                cmp_owned::check(cx, op.node, lhs, rhs);
                float_cmp::check(cx, e, op.node, lhs, rhs);
                modulo_one::check(cx, e, op.node, rhs);
                modulo_arithmetic::check(cx, e, op.node, lhs, rhs);
            },
            ExprKind::AssignOp(op, lhs, rhs) => {
                self.arithmetic_context.check_binary(cx, e, op.node, lhs, rhs);
                misrefactored_assign_op::check(cx, e, op.node, lhs, rhs);
                modulo_arithmetic::check(cx, e, op.node, lhs, rhs);
            },
            ExprKind::Assign(lhs, rhs, _) => {
                assign_op_pattern::check(cx, e, lhs, rhs);
                self_assignment::check(cx, e, lhs, rhs);
            },
            ExprKind::Unary(op, arg) => {
                if op == UnOp::Neg {
                    self.arithmetic_context.check_negate(cx, e, arg);
                }
            },
            _ => (),
        }
    }

    fn check_expr_post(&mut self, _: &LateContext<'_>, e: &Expr<'_>) {
        self.arithmetic_context.expr_post(e.hir_id);
    }

    fn check_body(&mut self, cx: &LateContext<'tcx>, b: &'tcx Body<'_>) {
        self.arithmetic_context.enter_body(cx, b);
    }

    fn check_body_post(&mut self, cx: &LateContext<'tcx>, b: &'tcx Body<'_>) {
        self.arithmetic_context.body_post(cx, b);
    }
}

fn macro_with_not_op(e: &Expr<'_>) -> bool {
    if let ExprKind::Unary(_, e) = e.kind {
        e.span.from_expansion()
    } else {
        false
    }
}
