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
    pub stmt: bool,

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
    pub leftmost_subexpression_in_stmt: bool,

    /// This is the difference between:
    ///
    /// ```ignore (illustrative)
    /// if let _ = (Struct {}) {}  // needs parens
    ///
    /// match () {
    ///     () if let _ = Struct {} => {}  // no parens
    /// }
    /// ```
    pub parenthesize_exterior_struct_lit: bool,
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
