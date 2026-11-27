#![feature(gca_min_const_items)]
#![feature(guard_patterns)]
#![feature(pin_ergonomics)]

#[cfg(false)]
fn check() {
    // Assign
    x = #[attr] y; //~ ERROR ambiguous expression attributes are unstable

    // AddrOf
    & #[attr] x + y; //~ ERROR ambiguous expression attributes are unstable
    &mut #[attr] x + y; //~ ERROR ambiguous expression attributes are unstable
    &raw const #[attr] x + y; //~ ERROR ambiguous expression attributes are unstable
    &raw mut #[attr] x + y; //~ ERROR ambiguous expression attributes are unstable
    &pin const #[attr] x + y; //~ ERROR ambiguous expression attributes are unstable
    &pin mut #[attr] x + y; //~ ERROR ambiguous expression attributes are unstable

    // Let
    if let x = #[attr] y {} //~ ERROR ambiguous expression attributes are unstable

    // Guard
    match 0 { _ if #[attr] x => y }; //~ ERROR ambiguous expression attributes are unstable
    for _ if #[attr] x in z {} //~ ERROR ambiguous expression attributes are unstable

    // AssignOp
    #[attr] x += y; //~ ERROR ambiguous expression attributes are unstable
    x += #[attr] y; //~ ERROR ambiguous expression attributes are unstable
    x += #[attr] y += z; //~ ERROR ambiguous expression attributes are unstable

    // Range
    #[attr] x .. y; //~ ERROR ambiguous expression attributes are unstable
    x .. #[attr] y; //~ ERROR ambiguous expression attributes are unstable
    #[attr] x ..; //~ ERROR ambiguous expression attributes are unstable
    #[attr] .. x; //~ ERROR attributes are not allowed on range expressions starting with `..`
    .. #[attr] x; //~ ERROR ambiguous expression attributes are unstable

    // Unary
    * #[attr] x + y; //~ ERROR ambiguous expression attributes are unstable
    ! #[attr] x + y; //~ ERROR ambiguous expression attributes are unstable
    - #[attr] x + y; //~ ERROR ambiguous expression attributes are unstable

    // Binary
    #[attr] x + y; //~ ERROR ambiguous expression attributes are unstable
    x + #[attr] y; //~ ERROR ambiguous expression attributes are unstable
    x + #[attr] y + z; //~ ERROR ambiguous expression attributes are unstable

    // PatKind::Expr
    match 0 { #[attr] const { x } => y } //~ ERROR const blocks cannot be used as patterns

    // PatKind::Range
    match 0 { (#[attr] x .. y) => {} } //~ ERROR expected pattern, found `#`
    match 0 { x .. #[attr] y => {} } //~ ERROR expected one of `=>`, `if`, or `|`, found `#`
    match 0 { (#[attr] x ..) => {} } //~ ERROR expected pattern, found `#`
    match 0 { (#[attr] .. x) => {} } //~ ERROR expected pattern, found `#`
    match 0 { .. #[attr] x => {} } //~ ERROR expected one of `=>`, `if`, or `|`, found `#`

    // AnonConst
    path::<#[attr] 1>; //~ ERROR attributes cannot be applied to generic arguments
    path::<#[attr] const { 1 }>; //~ ERROR attributes cannot be applied to generic arguments
    path::<#[attr] { 1 }>; //~ ERROR attributes cannot be applied to generic arguments
}

fn main() {}
