// Check that equality predicates get rejected.
// In the future we might support these or at least a restricted form of them
// where the LHS is a qpath. Presently however, type checking isn't impl'ed.
//
// See also: <https://github.com/rust-lang/rust/issues/20041>.

fn f() where u8 = u16 {}
//~^ ERROR equality constraints are not supported in where-clauses

fn g() where for<'a> &'static (u8,) == u16, {}
//~^ ERROR equality constraints are not supported in where-clauses

// Ensure that they're not just semantically invalid but also syntactically!
// This allows us to change the syntax all we like (we still need to decide
// between `=` and `==` for example) or even remove it entirely.
#[cfg(false)]
fn h<T: Iterator>()
where
    T::Item = fn(),
//~^ ERROR equality constraints are not supported in where-clauses
    <T as Iterator>::Item == for<> fn(),
//~^ ERROR equality constraints are not supported in where-clauses
{}

fn main() {}
