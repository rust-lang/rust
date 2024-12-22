use rustc_ast::util::classify;
use rustc_ast::util::parser::{self, ExprPrecedence};
use rustc_ast::{Expr, ExprKind, YieldKind};

// The default amount of fixing is minimal fixing, so all fixups are set to `false` by `Default`.
// Fixups should be turned on in a targeted fashion where needed.
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct FixupContext {
    /// Print expression such that it can be parsed back as a statement
    /// consisting of the original expression.
    ///
    /// The effect of this is for binary operators in statement position to set
    /// `leftmost_subexpression_in_stmt` when printing their left-hand operand.
    ///
    /// ```ignore (illustrative)
    /// (match x {}) - 1;  // match needs parens when LHS of binary operator
    ///
    /// match x {};  // not when its own statement
    /// ```
    stmt: bool,

    /// This is the difference between:
    ///
    /// ```ignore (illustrative)
    /// (match x {}) - 1;  // subexpression needs parens
    ///
    /// let _ = match x {} - 1;  // no parens
    /// ```
    ///
    /// There are 3 distinguishable contexts in which `print_expr` might be
    /// called with the expression `$match` as its argument, where `$match`
    /// represents an expression of kind `ExprKind::Match`:
    ///
    ///   - stmt=false leftmost_subexpression_in_stmt=false
    ///
    ///     Example: `let _ = $match - 1;`
    ///
    ///     No parentheses required.
    ///
    ///   - stmt=false leftmost_subexpression_in_stmt=true
    ///
    ///     Example: `$match - 1;`
    ///
    ///     Must parenthesize `($match)`, otherwise parsing back the output as a
    ///     statement would terminate the statement after the closing brace of
    ///     the match, parsing `-1;` as a separate statement.
    ///
    ///   - stmt=true leftmost_subexpression_in_stmt=false
    ///
    ///     Example: `$match;`
    ///
    ///     No parentheses required.
    leftmost_subexpression_in_stmt: bool,

    /// Print expression such that it can be parsed as a match arm.
    ///
    /// This is almost equivalent to `stmt`, but the grammar diverges a tiny bit
    /// between statements and match arms when it comes to braced macro calls.
    /// Macro calls with brace delimiter terminate a statement without a
    /// semicolon, but do not terminate a match-arm without comma.
    ///
    /// ```ignore (illustrative)
    /// m! {} - 1;  // two statements: a macro call followed by -1 literal
    ///
    /// match () {
    ///     _ => m! {} - 1,  // binary subtraction operator
    /// }
    /// ```
    match_arm: bool,

    /// This is almost equivalent to `leftmost_subexpression_in_stmt`, other
    /// than for braced macro calls.
    ///
    /// If we have `m! {} - 1` as an expression, the leftmost subexpression
    /// `m! {}` will need to be parenthesized in the statement case but not the
    /// match-arm case.
    ///
    /// ```ignore (illustrative)
    /// (m! {}) - 1;  // subexpression needs parens
    ///
    /// match () {
    ///     _ => m! {} - 1,  // no parens
    /// }
    /// ```
    leftmost_subexpression_in_match_arm: bool,

    /// This is the difference between:
    ///
    /// ```ignore (illustrative)
    /// if let _ = (Struct {}) {}  // needs parens
    ///
    /// match () {
    ///     () if let _ = Struct {} => {}  // no parens
    /// }
    /// ```
    parenthesize_exterior_struct_lit: bool,

    /// This is the difference between:
    ///
    /// ```ignore (illustrative)
    /// let _ = (return) - 1;  // without paren, this would return -1
    ///
    /// let _ = return + 1;  // no paren because '+' cannot begin expr
    /// ```
    next_operator_can_begin_expr: bool,

    /// This is the difference between:
    ///
    /// ```ignore (illustrative)
    /// let _ = 1 + return 1;  // no parens if rightmost subexpression
    ///
    /// let _ = 1 + (return 1) + 1;  // needs parens
    /// ```
    next_operator_can_continue_expr: bool,
}

impl FixupContext {
    /// Create the initial fixup for printing an expression in statement
    /// position.
    pub(crate) fn new_stmt() -> Self {
        FixupContext { stmt: true, ..FixupContext::default() }
    }

    /// Create the initial fixup for printing an expression as the right-hand
    /// side of a match arm.
    pub(crate) fn new_match_arm() -> Self {
        FixupContext { match_arm: true, ..FixupContext::default() }
    }

    /// Create the initial fixup for printing an expression as the "condition"
    /// of an `if` or `while`. There are a few other positions which are
    /// grammatically equivalent and also use this, such as the iterator
    /// expression in `for` and the scrutinee in `match`.
    pub(crate) fn new_cond() -> Self {
        FixupContext { parenthesize_exterior_struct_lit: true, ..FixupContext::default() }
    }

    /// Transform this fixup into the one that should apply when printing the
    /// leftmost subexpression of the current expression.
    ///
    /// The leftmost subexpression is any subexpression that has the same first
    /// token as the current expression, but has a different last token.
    ///
    /// For example in `$a + $b` and `$a.method()`, the subexpression `$a` is a
    /// leftmost subexpression.
    ///
    /// Not every expression has a leftmost subexpression. For example neither
    /// `-$a` nor `[$a]` have one.
    pub(crate) fn leftmost_subexpression(self) -> Self {
        FixupContext {
            stmt: false,
            leftmost_subexpression_in_stmt: self.stmt || self.leftmost_subexpression_in_stmt,
            match_arm: false,
            leftmost_subexpression_in_match_arm: self.match_arm
                || self.leftmost_subexpression_in_match_arm,
            next_operator_can_begin_expr: false,
            next_operator_can_continue_expr: true,
            ..self
        }
    }

    /// Transform this fixup into the one that should apply when printing a
    /// leftmost subexpression followed by a `.` or `?` token, which confer
    /// different statement boundary rules compared to other leftmost
    /// subexpressions.
    pub(crate) fn leftmost_subexpression_with_dot(self) -> Self {
        FixupContext {
            stmt: self.stmt || self.leftmost_subexpression_in_stmt,
            leftmost_subexpression_in_stmt: false,
            match_arm: self.match_arm || self.leftmost_subexpression_in_match_arm,
            leftmost_subexpression_in_match_arm: false,
            next_operator_can_begin_expr: false,
            next_operator_can_continue_expr: true,
            ..self
        }
    }

    /// Transform this fixup into the one that should apply when printing a
    /// leftmost subexpression followed by punctuation that is legal as the
    /// first token of an expression.
    pub(crate) fn leftmost_subexpression_with_operator(
        self,
        next_operator_can_begin_expr: bool,
    ) -> Self {
        FixupContext { next_operator_can_begin_expr, ..self.leftmost_subexpression() }
    }

    /// Transform this fixup into the one that should apply when printing the
    /// rightmost subexpression of the current expression.
    ///
    /// The rightmost subexpression is any subexpression that has a different
    /// first token than the current expression, but has the same last token.
    ///
    /// For example in `$a + $b` and `-$b`, the subexpression `$b` is a
    /// rightmost subexpression.
    ///
    /// Not every expression has a rightmost subexpression. For example neither
    /// `[$b]` nor `$a.f($b)` have one.
    pub(crate) fn rightmost_subexpression(self) -> Self {
        FixupContext {
            stmt: false,
            leftmost_subexpression_in_stmt: false,
            match_arm: false,
            leftmost_subexpression_in_match_arm: false,
            ..self
        }
    }

    /// Determine whether parentheses are needed around the given expression to
    /// head off an unintended statement boundary.
    ///
    /// The documentation on `FixupContext::leftmost_subexpression_in_stmt` has
    /// examples.
    pub(crate) fn would_cause_statement_boundary(self, expr: &Expr) -> bool {
        (self.leftmost_subexpression_in_stmt && !classify::expr_requires_semi_to_be_stmt(expr))
            || (self.leftmost_subexpression_in_match_arm && classify::expr_is_complete(expr))
    }

    /// Determine whether parentheses are needed around the given `let`
    /// scrutinee.
    ///
    /// In `if let _ = $e {}`, some examples of `$e` that would need parentheses
    /// are:
    ///
    ///   - `Struct {}.f()`, because otherwise the `{` would be misinterpreted
    ///     as the opening of the if's then-block.
    ///
    ///   - `true && false`, because otherwise this would be misinterpreted as a
    ///     "let chain".
    pub(crate) fn needs_par_as_let_scrutinee(self, expr: &Expr) -> bool {
        self.parenthesize_exterior_struct_lit && parser::contains_exterior_struct_lit(expr)
            || parser::needs_par_as_let_scrutinee(self.precedence(expr))
    }

    /// Determines the effective precedence of a subexpression. Some expressions
    /// have higher or lower precedence when adjacent to particular operators.
    pub(crate) fn precedence(self, expr: &Expr) -> ExprPrecedence {
        if self.next_operator_can_begin_expr {
            // Decrease precedence of value-less jumps when followed by an
            // operator that would otherwise get interpreted as beginning a
            // value for the jump.
            if let ExprKind::Break(..)
            | ExprKind::Ret(..)
            | ExprKind::Yeet(..)
            | ExprKind::Yield(YieldKind::Prefix(..)) = expr.kind
            {
                return ExprPrecedence::Jump;
            }
        }

        if !self.next_operator_can_continue_expr {
            // Increase precedence of expressions that extend to the end of
            // current statement or group.
            if let ExprKind::Break(..)
            | ExprKind::Closure(..)
            | ExprKind::Ret(..)
            | ExprKind::Yeet(..)
            | ExprKind::Yield(YieldKind::Prefix(..))
            | ExprKind::Range(None, ..) = expr.kind
            {
                return ExprPrecedence::Prefix;
            }
        }

        expr.precedence()
    }
}
