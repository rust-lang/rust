// Imports.

// Long import.
use syntax::ast::{ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic,
                  ItemDefaultImpl};

use {Foo, Bar};
use Foo::{Bar, Baz};
pub use syntax::ast::{Expr_, Expr, ExprAssign, ExprCall, ExprMethodCall,
                      ExprPath};

mod Foo {
    pub use syntax::ast::{Expr_, ExprEval, ToExpr, ExprMethodCall, ToExprPath};

    mod Foo2 {
        pub use syntax::ast::{Expr_, ExprEval, ToExpr, ExprMethodCall,
                              ToExprPath};
    }
}

fn test() {
    use Baz::*;
}
