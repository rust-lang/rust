
extern crate clippy;

use clippy::consts;
use syntax::ast::*;

#[test]
fn test_lit() {
    assert_eq!(ConstantBool(true), constant(&Context,
        Expr{ node_id: 1, node: ExprLit(LitBool(true)), span: default() }));
}
