use rustc_ast::util::{classify, parser};
use rustc_ast::Expr;

#[derive(Copy, Clone, Debug)]
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
}

/// The default amount of fixing is minimal fixing. Fixups should be turned on
/// in a targeted fashion where needed.
impl Default for FixupContext {
    fn default() -> Self {
        FixupContext {
            stmt: false,
            leftmost_subexpression_in_stmt: false,
            parenthesize_exterior_struct_lit: false,
        }
    }
}

impl FixupContext {
    /// Create the initial fixup for printing an expression in statement
    /// position.
    ///
    /// This is currently also used for printing an expression as a match-arm,
    /// but this is incorrect and leads to over-parenthesizing.
    pub fn new_stmt() -> Self {
        FixupContext { stmt: true, ..FixupContext::default() }
    }

    /// Create the initial fixup for printing an expression as the "condition"
    /// of an `if` or `while`. There are a few other positions which are
    /// grammatically equivalent and also use this, such as the iterator
    /// expression in `for` and the scrutinee in `match`.
    pub fn new_cond() -> Self {
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
    pub fn leftmost_subexpression(self) -> Self {
        FixupContext {
            stmt: false,
            leftmost_subexpression_in_stmt: self.stmt || self.leftmost_subexpression_in_stmt,
            ..self
        }
    }

    /// Transform this fixup into the one that should apply when printing any
    /// subexpression that is neither a leftmost subexpression nor surrounded in
    /// delimiters.
    ///
    /// This is for any subexpression that has a different first token than the
    /// current expression, and is not surrounded by a paren/bracket/brace. For
    /// example the `$b` in `$a + $b` and `-$b`, but not the one in `[$b]` or
    /// `$a.f($b)`.
    pub fn subsequent_subexpression(self) -> Self {
        FixupContext { stmt: false, leftmost_subexpression_in_stmt: false, ..self }
    }

    /// Determine whether parentheses are needed around the given expression to
    /// head off an unintended statement boundary.
    ///
    /// The documentation on `FixupContext::leftmost_subexpression_in_stmt` has
    /// examples.
    pub fn would_cause_statement_boundary(self, expr: &Expr) -> bool {
        self.leftmost_subexpression_in_stmt
            && match expr.kind {
                ExprKind::MacCall(_) => false,
                _ => !classify::expr_requires_semi_to_be_stmt(expr),
            }
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
    pub fn needs_par_as_let_scrutinee(self, expr: &Expr) -> bool {
        self.parenthesize_exterior_struct_lit && parser::contains_exterior_struct_lit(expr)
            || parser::needs_par_as_let_scrutinee(expr.precedence().order())
    }
}
