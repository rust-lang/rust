// rustfmt-normalize_comments: true

// Imports.

// Long import.
use rustc_ast::ast::{ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic, ItemDefaultImpl};
use exceedingly::looooooooooooooooooooooooooooooooooooooooooooooooooooooooooong::import::path::{ItemA, ItemB};
use exceedingly::loooooooooooooooooooooooooooooooooooooooooooooooooooooooong::import::path::{ItemA, ItemB};

use list::{
    // Some item
    SomeItem /* Comment */, /* Another item */ AnotherItem /* Another Comment */, // Last Item
    LastItem
};

use test::{  Other          /* C   */  , /*   A   */ self  /*    B     */    };

use rustc_ast::{self};
use {/* Pre-comment! */
     Foo, Bar /* comment */};
use Foo::{Bar, Baz};
pub use rustc_ast::ast::{Expr_, Expr, ExprAssign, ExprCall, ExprMethodCall, ExprPath};

use rustc_ast::some::{};

use self;
use std::io::{self};
use std::io::self;

mod Foo {
    pub use rustc_ast::ast::{
        ItemForeignMod,
        ItemImpl,
        ItemMac,
        ItemMod,
        ItemStatic,
        ItemDefaultImpl
    };

    mod Foo2 {
        pub use rustc_ast::ast::{ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic, self, ItemDefaultImpl};
    }
}

fn test() {
use Baz::*;
        use Qux;
}

// Simple imports
use  foo::bar::baz as baz ;
use bar::quux  as    kaas;
use  foo;

// With aliases.
use foo::{self as bar, baz};
use foo::{self as bar};
use foo::{qux as bar};
use foo::{baz, qux as bar};

// With absolute paths
use ::foo;
use ::foo::{Bar};
use ::foo::{Bar, Baz};
use ::{Foo};
use ::{Bar, Baz};

// Root globs
use *;
use ::*;

// spaces used to cause glob imports to disappear (#1356)
use super:: * ;
use foo::issue_1356:: * ;

// We shouldn't remove imports which have attributes attached (#1858)
#[cfg(unix)]
use self::unix::{};

// nested imports
use foo::{a, bar::{baz, qux, xxxxxxxxxxx, yyyyyyyyyyyyy, zzzzzzzzzzzzzzzz, foo::{a, b, cxxxxxxxxxxxxx, yyyyyyyyyyyyyy, zzzzzzzzzzzzzzzz}}, b, boo, c,};

use fooo::{baar::{foobar::{xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx, yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy, zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz}}, z, bar, bar::*, x, y};

use exonum::{api::{Api, ApiError}, blockchain::{self, BlockProof, Blockchain, Transaction, TransactionSet}, crypto::{Hash, PublicKey}, helpers::Height, node::TransactionSend, storage::{ListProof, MapProof}};

// nested imports with a single sub-tree.
use a::{b::{c::*}};
use a::{b::{c::{}}};
use a::{b::{c::d}};
use a::{b::{c::{xxx, yyy, zzz}}};

// #2645
/// This line is not affected.
// This line is deleted.
use c;

// #2670
#[macro_use]
use imports_with_attr;

// #2888
use std::f64::consts::{SQRT_2, E, PI};

// #3273
#[rustfmt::skip]
use std::fmt::{self, {Display, Formatter}};
