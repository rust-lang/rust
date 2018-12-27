//! Builds MIR from expressions. As a caller into this module, you
//! have many options, but the first thing you have to decide is
//! whether you are evaluating this expression for its *value*, its
//! *location*, or as a *constant*.
//!
//! Typically, you want the value: e.g., if you are doing `expr_a +
//! expr_b`, you want the values of those expressions. In that case,
//! you want one of the following functions. Note that if the expr has
//! a type that is not `Copy`, then using any of these functions will
//! "move" the value out of its current home (if any).
//!
//! - `into` -- writes the value into a specific location, which
//!   should be uninitialized
//! - `as_operand` -- evaluates the value and yields an `Operand`,
//!   suitable for use as an argument to an `Rvalue`
//! - `as_temp` -- evaluates into a temporary; this is similar to `as_operand`
//!   except it always returns a fresh place, even for constants
//! - `as_rvalue` -- yields an `Rvalue`, suitable for use in an assignment;
//!   as of this writing, never needed outside of the `expr` module itself
//!
//! Sometimes though want the expression's *location*. An example
//! would be during a match statement, or the operand of the `&`
//! operator. In that case, you want `as_place`. This will create a
//! temporary if necessary.
//!
//! Finally, if it's a constant you seek, then call
//! `as_constant`. This creates a `Constant<H>`, but naturally it can
//! only be used on constant expressions and hence is needed only in
//! very limited contexts.
//!
//! ### Implementation notes
//!
//! For any given kind of expression, there is generally one way that
//! can be lowered most naturally. This is specified by the
//! `Category::of` function in the `category` module. For example, a
//! struct expression (or other expression that creates a new value)
//! is typically easiest to write in terms of `as_rvalue` or `into`,
//! whereas a reference to a field is easiest to write in terms of
//! `as_place`. (The exception to this is scope and paren
//! expressions, which have no category.)
//!
//! Therefore, the various functions above make use of one another in
//! a descending fashion. For any given expression, you should pick
//! the most suitable spot to implement it, and then just let the
//! other fns cycle around. The handoff works like this:
//!
//! - `into(place)` -> fallback is to create a rvalue with `as_rvalue` and assign it to `place`
//! - `as_rvalue` -> fallback is to create an Operand with `as_operand` and use `Rvalue::use`
//! - `as_operand` -> either invokes `as_constant` or `as_temp`
//! - `as_constant` -> (no fallback)
//! - `as_temp` -> creates a temporary and either calls `as_place` or `into`
//! - `as_place` -> for rvalues, falls back to `as_temp` and returns that
//!
//! As you can see, there is a cycle where `into` can (in theory) fallback to `as_temp`
//! which can fallback to `into`. So if one of the `ExprKind` variants is not, in fact,
//! implemented in the category where it is supposed to be, there will be a problem.
//!
//! Of those fallbacks, the most interesting one is `into`, because
//! it discriminates based on the category of the expression. This is
//! basically the point where the "by value" operations are bridged
//! over to the "by reference" mode (`as_place`).

mod as_constant;
mod as_operand;
mod as_place;
mod as_rvalue;
mod as_temp;
mod category;
mod into;
mod stmt;
