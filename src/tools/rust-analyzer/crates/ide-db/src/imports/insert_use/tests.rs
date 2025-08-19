use stdx::trim_indent;
use test_fixture::WithFixture;
use test_utils::{CURSOR_MARKER, assert_eq_text};

use super::*;

#[test]
fn trailing_comment_in_empty_file() {
    check(
        "foo::bar",
        r#"
struct Struct;
// 0 = 1
"#,
        r#"
use foo::bar;

struct Struct;
// 0 = 1
"#,
        ImportGranularity::Crate,
    );
}

#[test]
fn respects_cfg_attr_fn_body() {
    check(
        r"bar::Bar",
        r#"
#[cfg(test)]
fn foo() {$0}
"#,
        r#"
#[cfg(test)]
fn foo() {
    use bar::Bar;
}
"#,
        ImportGranularity::Crate,
    );
}

#[test]
fn respects_cfg_attr_fn_sig() {
    check(
        r"bar::Bar",
        r#"
#[cfg(test)]
fn foo($0) {}
"#,
        r#"
#[cfg(test)]
use bar::Bar;

#[cfg(test)]
fn foo() {}
"#,
        ImportGranularity::Crate,
    );
}

#[test]
fn respects_cfg_attr_const() {
    check(
        r"bar::Bar",
        r#"
#[cfg(test)]
const FOO: Bar = {$0};
"#,
        r#"
#[cfg(test)]
const FOO: Bar = {
    use bar::Bar;
};
"#,
        ImportGranularity::Crate,
    );
}

#[test]
fn respects_cfg_attr_impl() {
    check(
        r"bar::Bar",
        r#"
#[cfg(test)]
impl () {$0}
"#,
        r#"
#[cfg(test)]
use bar::Bar;

#[cfg(test)]
impl () {}
"#,
        ImportGranularity::Crate,
    );
}

#[test]
fn respects_cfg_attr_multiple_layers() {
    check(
        r"bar::Bar",
        r#"
#[cfg(test)]
impl () {
    #[cfg(test2)]
    fn f($0) {}
}
"#,
        r#"
#[cfg(test)]
#[cfg(test2)]
use bar::Bar;

#[cfg(test)]
impl () {
    #[cfg(test2)]
    fn f() {}
}
"#,
        ImportGranularity::Crate,
    );
}

#[test]
fn insert_skips_lone_glob_imports() {
    check(
        "use foo::baz::A",
        r"
use foo::bar::*;
",
        r"
use foo::bar::*;
use foo::baz::A;
",
        ImportGranularity::Crate,
    );
}

#[test]
fn insert_not_group() {
    cov_mark::check!(insert_no_grouping_last);
    check_with_config(
        "use external_crate2::bar::A",
        r"
use std::bar::B;
use external_crate::bar::A;
use crate::bar::A;
use self::bar::A;
use super::bar::A;",
        r"
use std::bar::B;
use external_crate::bar::A;
use crate::bar::A;
use self::bar::A;
use super::bar::A;
use external_crate2::bar::A;",
        &InsertUseConfig {
            granularity: ImportGranularity::Item,
            enforce_granularity: true,
            prefix_kind: PrefixKind::Plain,
            group: false,
            skip_glob_imports: true,
        },
    );
}

#[test]
fn insert_existing() {
    check_crate("std::fs", "use std::fs;", "use std::fs;")
}

#[test]
fn insert_start() {
    check_none(
        "std::bar::AA",
        r"
use std::bar::B;
use std::bar::D;
use std::bar::F;
use std::bar::G;",
        r"
use std::bar::AA;
use std::bar::B;
use std::bar::D;
use std::bar::F;
use std::bar::G;",
    )
}

#[test]
fn insert_start_indent() {
    check_none(
        "std::bar::AA",
        r"
    use std::bar::B;
    use std::bar::C;",
        r"
    use std::bar::AA;
    use std::bar::B;
    use std::bar::C;",
    );
    check_none(
        "std::bar::r#AA",
        r"
    use std::bar::B;
    use std::bar::C;",
        r"
    use std::bar::r#AA;
    use std::bar::B;
    use std::bar::C;",
    );
}

#[test]
fn insert_middle() {
    cov_mark::check!(insert_group);
    check_none(
        "std::bar::EE",
        r"
use std::bar::A;
use std::bar::D;
use std::bar::F;
use std::bar::G;",
        r"
use std::bar::A;
use std::bar::D;
use std::bar::EE;
use std::bar::F;
use std::bar::G;",
    )
}

#[test]
fn insert_middle_indent() {
    check_none(
        "std::bar::EE",
        r"
    use std::bar::A;
    use std::bar::D;
    use std::bar::F;
    use std::bar::G;",
        r"
    use std::bar::A;
    use std::bar::D;
    use std::bar::EE;
    use std::bar::F;
    use std::bar::G;",
    );
    check_none(
        "std::bar::r#EE",
        r"
    use std::bar::A;
    use std::bar::D;
    use std::bar::F;
    use std::bar::G;",
        r"
    use std::bar::A;
    use std::bar::D;
    use std::bar::r#EE;
    use std::bar::F;
    use std::bar::G;",
    );
}

#[test]
fn insert_end() {
    cov_mark::check!(insert_group_last);
    check_none(
        "std::bar::ZZ",
        r"
use std::bar::A;
use std::bar::D;
use std::bar::F;
use std::bar::G;",
        r"
use std::bar::A;
use std::bar::D;
use std::bar::F;
use std::bar::G;
use std::bar::ZZ;",
    )
}

#[test]
fn insert_end_indent() {
    check_none(
        "std::bar::ZZ",
        r"
    use std::bar::A;
    use std::bar::D;
    use std::bar::F;
    use std::bar::G;",
        r"
    use std::bar::A;
    use std::bar::D;
    use std::bar::F;
    use std::bar::G;
    use std::bar::ZZ;",
    );
    check_none(
        "std::bar::r#ZZ",
        r"
    use std::bar::A;
    use std::bar::D;
    use std::bar::F;
    use std::bar::G;",
        r"
    use std::bar::A;
    use std::bar::D;
    use std::bar::F;
    use std::bar::G;
    use std::bar::r#ZZ;",
    );
}

#[test]
fn insert_middle_nested() {
    check_none(
        "std::bar::EE",
        r"
use std::bar::A;
use std::bar::{D, Z}; // example of weird imports due to user
use std::bar::F;
use std::bar::G;",
        r"
use std::bar::A;
use std::bar::EE;
use std::bar::{D, Z}; // example of weird imports due to user
use std::bar::F;
use std::bar::G;",
    );
    check_none(
        "std::bar::r#EE",
        r"
use std::bar::A;
use std::bar::{D, Z}; // example of weird imports due to user
use std::bar::F;
use std::bar::G;",
        r"
use std::bar::A;
use std::bar::r#EE;
use std::bar::{D, Z}; // example of weird imports due to user
use std::bar::F;
use std::bar::G;",
    );
}

#[test]
fn insert_middle_groups() {
    check_none(
        "foo::bar::GG",
        r"
    use std::bar::A;
    use std::bar::D;

    use foo::bar::F;
    use foo::bar::H;",
        r"
    use std::bar::A;
    use std::bar::D;

    use foo::bar::F;
    use foo::bar::GG;
    use foo::bar::H;",
    )
}

#[test]
fn insert_first_matching_group() {
    check_none(
        "foo::bar::GG",
        r"
    use foo::bar::A;
    use foo::bar::D;

    use std;

    use foo::bar::F;
    use foo::bar::H;",
        r"
    use foo::bar::A;
    use foo::bar::D;
    use foo::bar::GG;

    use std;

    use foo::bar::F;
    use foo::bar::H;",
    )
}

#[test]
fn insert_missing_group_std() {
    cov_mark::check!(insert_group_new_group);
    check_none(
        "std::fmt",
        r"
    use foo::bar::A;
    use foo::bar::D;",
        r"
    use std::fmt;

    use foo::bar::A;
    use foo::bar::D;",
    )
}

#[test]
fn insert_missing_group_self() {
    cov_mark::check!(insert_group_no_group);
    check_none(
        "self::fmt",
        r"
use foo::bar::A;
use foo::bar::D;",
        r"
use foo::bar::A;
use foo::bar::D;

use self::fmt;",
    )
}

#[test]
fn insert_no_imports() {
    check_crate(
        "foo::bar",
        "fn main() {}",
        r"use foo::bar;

fn main() {}",
    )
}

#[test]
fn insert_empty_file() {
    cov_mark::check_count!(insert_empty_file, 2);

    // Default configuration
    // empty files will get two trailing newlines
    // this is due to the test case insert_no_imports above
    check_crate(
        "foo::bar",
        "",
        r"use foo::bar;

",
    );

    // "not group" configuration
    check_with_config(
        "use external_crate2::bar::A",
        r"",
        r"use external_crate2::bar::A;

",
        &InsertUseConfig {
            granularity: ImportGranularity::Item,
            enforce_granularity: true,
            prefix_kind: PrefixKind::Plain,
            group: false,
            skip_glob_imports: true,
        },
    );
}

#[test]
fn insert_empty_module() {
    cov_mark::check_count!(insert_empty_module, 2);

    // Default configuration
    check(
        "foo::bar",
        r"
mod x {$0}
",
        r"
mod x {
    use foo::bar;
}
",
        ImportGranularity::Item,
    );

    // "not group" configuration
    check_with_config(
        "foo::bar",
        r"mod x {$0}",
        r"mod x {
    use foo::bar;
}",
        &InsertUseConfig {
            granularity: ImportGranularity::Item,
            enforce_granularity: true,
            prefix_kind: PrefixKind::Plain,
            group: false,
            skip_glob_imports: true,
        },
    );
}

#[test]
fn insert_after_inner_attr() {
    cov_mark::check_count!(insert_empty_inner_attr, 2);

    // Default configuration
    check_crate(
        "foo::bar",
        r"#![allow(unused_imports)]",
        r"#![allow(unused_imports)]

use foo::bar;",
    );

    // "not group" configuration
    check_with_config(
        "foo::bar",
        r"#![allow(unused_imports)]",
        r"#![allow(unused_imports)]

use foo::bar;",
        &InsertUseConfig {
            granularity: ImportGranularity::Item,
            enforce_granularity: true,
            prefix_kind: PrefixKind::Plain,
            group: false,
            skip_glob_imports: true,
        },
    );
}

#[test]
fn insert_after_inner_attr2() {
    check_crate(
        "foo::bar",
        r"#![allow(unused_imports)]

#![no_std]
fn main() {}",
        r"#![allow(unused_imports)]

#![no_std]

use foo::bar;
fn main() {}",
    );
}

#[test]
fn inserts_after_single_line_inner_comments() {
    check_none(
        "foo::bar::Baz",
        "//! Single line inner comments do not allow any code before them.",
        r#"//! Single line inner comments do not allow any code before them.

use foo::bar::Baz;"#,
    );
    check_none(
        "foo::bar::Baz",
        r"mod foo {
    //! Single line inner comments do not allow any code before them.
$0
}",
        r"mod foo {
    //! Single line inner comments do not allow any code before them.

    use foo::bar::Baz;

}",
    );
}

#[test]
fn inserts_after_single_line_comments() {
    check_none(
        "foo::bar::Baz",
        "// Represents a possible license header and/or general module comments",
        r#"// Represents a possible license header and/or general module comments

use foo::bar::Baz;"#,
    );
}

#[test]
fn inserts_after_shebang() {
    check_none(
        "foo::bar::Baz",
        "#!/usr/bin/env rust",
        r#"#!/usr/bin/env rust

use foo::bar::Baz;"#,
    );
}

#[test]
fn inserts_after_multiple_single_line_comments() {
    check_none(
        "foo::bar::Baz",
        "// Represents a possible license header and/or general module comments
// Second single-line comment
// Third single-line comment",
        r#"// Represents a possible license header and/or general module comments
// Second single-line comment
// Third single-line comment

use foo::bar::Baz;"#,
    );
}

#[test]
fn inserts_before_single_line_item_comments() {
    check_none(
        "foo::bar::Baz",
        r#"// Represents a comment about a function
fn foo() {}"#,
        r#"use foo::bar::Baz;

// Represents a comment about a function
fn foo() {}"#,
    );
}

#[test]
fn inserts_after_single_line_header_comments_and_before_item() {
    check_none(
        "foo::bar::Baz",
        r#"// Represents a possible license header
// Line two of possible license header

fn foo() {}"#,
        r#"// Represents a possible license header
// Line two of possible license header

use foo::bar::Baz;

fn foo() {}"#,
    );
}

#[test]
fn inserts_after_multiline_inner_comments() {
    check_none(
        "foo::bar::Baz",
        r#"/*! Multiline inner comments do not allow any code before them. */

/*! Still an inner comment, cannot place any code before. */
fn main() {}"#,
        r#"/*! Multiline inner comments do not allow any code before them. */

/*! Still an inner comment, cannot place any code before. */

use foo::bar::Baz;
fn main() {}"#,
    )
}

#[test]
fn inserts_after_all_inner_items() {
    check_none(
        "foo::bar::Baz",
        r#"#![allow(unused_imports)]
/*! Multiline line comment 2 */


//! Single line comment 1
#![no_std]
//! Single line comment 2
fn main() {}"#,
        r#"#![allow(unused_imports)]
/*! Multiline line comment 2 */


//! Single line comment 1
#![no_std]
//! Single line comment 2

use foo::bar::Baz;
fn main() {}"#,
    )
}

#[test]
fn merge_groups() {
    check_module("std::io", r"use std::fmt;", r"use std::{fmt, io};");
    check_one("std::io", r"use {std::fmt};", r"use {std::{fmt, io}};");
    check_one("std::io", r"use std::fmt;", r"use {std::{fmt, io}};");
}

#[test]
fn merge_groups_last() {
    check_module(
        "std::io",
        r"use std::fmt::{Result, Display};",
        r"use std::fmt::{Result, Display};
use std::io;",
    );
    check_one(
        "std::io",
        r"use {std::fmt::{Result, Display}};",
        r"use {std::{fmt::{Display, Result}, io}};",
    );
}

#[test]
fn merge_last_into_self() {
    check_module("foo::bar::baz", r"use foo::bar;", r"use foo::bar::{self, baz};");
    check_one("foo::bar::baz", r"use {foo::bar};", r"use {foo::bar::{self, baz}};");
}

#[test]
fn merge_groups_full() {
    check_crate(
        "std::io",
        r"use std::fmt::{Result, Display};",
        r"use std::{fmt::{Display, Result}, io};",
    );
    check_one(
        "std::io",
        r"use {std::fmt::{Result, Display}};",
        r"use {std::{fmt::{Display, Result}, io}};",
    );
}

#[test]
fn merge_groups_long_full() {
    check_crate(
        "std::foo::bar::Baz",
        r"use std::foo::bar::Qux;",
        r"use std::foo::bar::{Baz, Qux};",
    );
    check_crate(
        "std::foo::bar::r#Baz",
        r"use std::foo::bar::Qux;",
        r"use std::foo::bar::{r#Baz, Qux};",
    );
    check_one(
        "std::foo::bar::Baz",
        r"use {std::foo::bar::Qux};",
        r"use {std::foo::bar::{Baz, Qux}};",
    );
}

#[test]
fn merge_groups_long_last() {
    check_module(
        "std::foo::bar::Baz",
        r"use std::foo::bar::Qux;",
        r"use std::foo::bar::{Baz, Qux};",
    )
}

#[test]
fn merge_groups_long_full_list() {
    check_crate(
        "std::foo::bar::Baz",
        r"use std::foo::bar::{Qux, Quux};",
        r"use std::foo::bar::{Baz, Quux, Qux};",
    );
    check_crate(
        "std::foo::bar::r#Baz",
        r"use std::foo::bar::{Qux, Quux};",
        r"use std::foo::bar::{r#Baz, Quux, Qux};",
    );
    check_one(
        "std::foo::bar::Baz",
        r"use {std::foo::bar::{Qux, Quux}};",
        r"use {std::foo::bar::{Baz, Quux, Qux}};",
    );
}

#[test]
fn merge_groups_long_last_list() {
    check_module(
        "std::foo::bar::Baz",
        r"use std::foo::bar::{Qux, Quux};",
        r"use std::foo::bar::{Baz, Quux, Qux};",
    )
}

#[test]
fn merge_groups_long_full_nested() {
    check_crate(
        "std::foo::bar::Baz",
        r"use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
        r"use std::foo::bar::{quux::{Fez, Fizz}, Baz, Qux};",
    );
    check_crate(
        "std::foo::bar::r#Baz",
        r"use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
        r"use std::foo::bar::{quux::{Fez, Fizz}, r#Baz, Qux};",
    );
    check_one(
        "std::foo::bar::Baz",
        r"use {std::foo::bar::{Qux, quux::{Fez, Fizz}}};",
        r"use {std::foo::bar::{quux::{Fez, Fizz}, Baz, Qux}};",
    );
}

#[test]
fn merge_groups_long_last_nested() {
    check_module(
        "std::foo::bar::Baz",
        r"use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
        r"use std::foo::bar::Baz;
use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
    )
}

#[test]
fn merge_groups_full_nested_deep() {
    check_crate(
        "std::foo::bar::quux::Baz",
        r"use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
        r"use std::foo::bar::{quux::{Baz, Fez, Fizz}, Qux};",
    );
    check_one(
        "std::foo::bar::quux::Baz",
        r"use {std::foo::bar::{Qux, quux::{Fez, Fizz}}};",
        r"use {std::foo::bar::{quux::{Baz, Fez, Fizz}, Qux}};",
    );
}

#[test]
fn merge_groups_full_nested_long() {
    check_crate(
        "std::foo::bar::Baz",
        r"use std::{foo::bar::Qux};",
        r"use std::foo::bar::{Baz, Qux};",
    );
}

#[test]
fn merge_groups_last_nested_long() {
    check_crate(
        "std::foo::bar::Baz",
        r"use std::{foo::bar::Qux};",
        r"use std::foo::bar::{Baz, Qux};",
    );
    check_one(
        "std::foo::bar::Baz",
        r"use {std::{foo::bar::Qux}};",
        r"use {std::foo::bar::{Baz, Qux}};",
    );
}

#[test]
fn merge_groups_skip_pub() {
    check_crate(
        "std::io",
        r"pub use std::fmt::{Result, Display};",
        r"pub use std::fmt::{Result, Display};
use std::io;",
    );
    check_one(
        "std::io",
        r"pub use {std::fmt::{Result, Display}};",
        r"pub use {std::fmt::{Result, Display}};
use {std::io};",
    );
}

#[test]
fn merge_groups_skip_pub_crate() {
    check_crate(
        "std::io",
        r"pub(crate) use std::fmt::{Result, Display};",
        r"pub(crate) use std::fmt::{Result, Display};
use std::io;",
    );
    check_one(
        "std::io",
        r"pub(crate) use {std::fmt::{Result, Display}};",
        r"pub(crate) use {std::fmt::{Result, Display}};
use {std::io};",
    );
}

#[test]
fn merge_groups_cfg_vs_no_cfg() {
    check_crate(
        "std::io",
        r#"
#[cfg(feature = "gated")] use std::fmt::{Result, Display};
"#,
        r#"
#[cfg(feature = "gated")] use std::fmt::{Result, Display};
use std::io;
"#,
    );
    check_one(
        "std::io",
        r#"
#[cfg(feature = "gated")] use {std::fmt::{Result, Display}};
"#,
        r#"
#[cfg(feature = "gated")] use {std::fmt::{Result, Display}};
use {std::io};
"#,
    );
}

#[test]
fn merge_groups_cfg_matching() {
    check_crate(
        "std::io",
        r#"
#[cfg(feature = "gated")] use std::fmt::{Result, Display};

#[cfg(feature = "gated")]
fn f($0) {}
"#,
        r#"
#[cfg(feature = "gated")] use std::{fmt::{Display, Result}, io};

#[cfg(feature = "gated")]
fn f() {}
"#,
    );
}

#[test]
fn split_out_merge() {
    // FIXME: This is suboptimal, we want to get `use std::fmt::{self, Result}`
    // instead.
    check_module(
        "std::fmt::Result",
        r"use std::{fmt, io};",
        r"use std::fmt::Result;
use std::{fmt, io};",
    )
}

#[test]
fn merge_into_module_import() {
    check_crate("std::fmt::Result", r"use std::{fmt, io};", r"use std::{fmt::{self, Result}, io};")
}

#[test]
fn merge_groups_self() {
    check_crate("std::fmt::Debug", r"use std::fmt;", r"use std::fmt::{self, Debug};")
}

#[test]
fn merge_mod_into_glob() {
    check_with_config(
        "token::TokenKind",
        r"use token::TokenKind::*;",
        r"use token::TokenKind::{self, *};",
        &InsertUseConfig {
            granularity: ImportGranularity::Crate,
            enforce_granularity: true,
            prefix_kind: PrefixKind::Plain,
            group: false,
            skip_glob_imports: false,
        },
    )
}

#[test]
fn merge_self_glob() {
    check_with_config(
        "self",
        r"use self::*;",
        r"use self::{self, *};",
        &InsertUseConfig {
            granularity: ImportGranularity::Crate,
            enforce_granularity: true,
            prefix_kind: PrefixKind::Plain,
            group: false,
            skip_glob_imports: false,
        },
    )
}

#[test]
fn merge_glob() {
    check_crate(
        "syntax::SyntaxKind",
        r"
use syntax::{SyntaxKind::*};",
        r"
use syntax::SyntaxKind::{self, *};",
    )
}

#[test]
fn merge_glob_nested() {
    check_crate(
        "foo::bar::quux::Fez",
        r"use foo::bar::{Baz, quux::*};",
        r"use foo::bar::{quux::{Fez, *}, Baz};",
    )
}

#[test]
fn merge_nested_considers_first_segments() {
    check_crate(
        "hir_ty::display::write_bounds_like_dyn_trait",
        r"use hir_ty::{autoderef, display::{HirDisplayError, HirFormatter}, method_resolution};",
        r"use hir_ty::{autoderef, display::{write_bounds_like_dyn_trait, HirDisplayError, HirFormatter}, method_resolution};",
    );
}

#[test]
fn skip_merge_last_too_long() {
    check_module(
        "foo::bar",
        r"use foo::bar::baz::Qux;",
        r"use foo::bar;
use foo::bar::baz::Qux;",
    );
}

#[test]
fn skip_merge_last_too_long2() {
    check_module(
        "foo::bar::baz::Qux",
        r"use foo::bar;",
        r"use foo::bar;
use foo::bar::baz::Qux;",
    );
}

#[test]
fn insert_short_before_long() {
    check_none(
        "foo::bar",
        r"use foo::bar::baz::Qux;",
        r"use foo::bar;
use foo::bar::baz::Qux;",
    );
}

#[test]
fn merge_last_fail() {
    check_merge_only_fail(
        r"use foo::bar::{baz::{Qux, Fez}};",
        r"use foo::bar::{baaz::{Quux, Feez}};",
        MergeBehavior::Module,
    );
}

#[test]
fn merge_last_fail1() {
    check_merge_only_fail(
        r"use foo::bar::{baz::{Qux, Fez}};",
        r"use foo::bar::baaz::{Quux, Feez};",
        MergeBehavior::Module,
    );
}

#[test]
fn merge_last_fail2() {
    check_merge_only_fail(
        r"use foo::bar::baz::{Qux, Fez};",
        r"use foo::bar::{baaz::{Quux, Feez}};",
        MergeBehavior::Module,
    );
}

#[test]
fn merge_last_fail3() {
    check_merge_only_fail(
        r"use foo::bar::baz::{Qux, Fez};",
        r"use foo::bar::baaz::{Quux, Feez};",
        MergeBehavior::Module,
    );
}

#[test]
fn guess_empty() {
    check_guess("", ImportGranularityGuess::Unknown);
}

#[test]
fn guess_single() {
    check_guess(r"use foo::{baz::{qux, quux}, bar};", ImportGranularityGuess::Crate);
    check_guess(r"use foo::bar;", ImportGranularityGuess::Unknown);
    check_guess(r"use foo::bar::{baz, qux};", ImportGranularityGuess::CrateOrModule);
    check_guess(r"use {foo::bar};", ImportGranularityGuess::One);
}

#[test]
fn guess_unknown() {
    check_guess(
        r"
use foo::bar::baz;
use oof::rab::xuq;
",
        ImportGranularityGuess::Unknown,
    );
}

#[test]
fn guess_item() {
    check_guess(
        r"
use foo::bar::baz;
use foo::bar::qux;
",
        ImportGranularityGuess::Item,
    );
}

#[test]
fn guess_module_or_item() {
    check_guess(
        r"
use foo::bar::Bar;
use foo::qux;
",
        ImportGranularityGuess::ModuleOrItem,
    );
    check_guess(
        r"
use foo::bar::Bar;
use foo::bar;
",
        ImportGranularityGuess::ModuleOrItem,
    );
}

#[test]
fn guess_module() {
    check_guess(
        r"
use foo::bar::baz;
use foo::bar::{qux, quux};
",
        ImportGranularityGuess::Module,
    );
    // this is a rather odd case, technically this file isn't following any style properly.
    check_guess(
        r"
use foo::bar::baz;
use foo::{baz::{qux, quux}, bar};
",
        ImportGranularityGuess::Module,
    );
    check_guess(
        r"
use foo::bar::Bar;
use foo::baz::Baz;
use foo::{Foo, Qux};
",
        ImportGranularityGuess::Module,
    );
}

#[test]
fn guess_crate_or_module() {
    check_guess(
        r"
use foo::bar::baz;
use oof::bar::{qux, quux};
",
        ImportGranularityGuess::CrateOrModule,
    );
}

#[test]
fn guess_crate() {
    check_guess(
        r"
use frob::bar::baz;
use foo::{baz::{qux, quux}, bar};
",
        ImportGranularityGuess::Crate,
    );
}

#[test]
fn guess_one() {
    check_guess(
        r"
use {
    frob::bar::baz,
    foo::{baz::{qux, quux}, bar}
};
",
        ImportGranularityGuess::One,
    );
}

#[test]
fn guess_skips_differing_vis() {
    check_guess(
        r"
use foo::bar::baz;
pub use foo::bar::qux;
",
        ImportGranularityGuess::Unknown,
    );
}

#[test]
fn guess_one_differing_vis() {
    check_guess(
        r"
use {foo::bar::baz};
pub use {foo::bar::qux};
",
        ImportGranularityGuess::One,
    );
}

#[test]
fn guess_skips_multiple_one_style_same_vis() {
    check_guess(
        r"
use {foo::bar::baz};
use {foo::bar::qux};
",
        ImportGranularityGuess::Unknown,
    );
}

#[test]
fn guess_skips_differing_attrs() {
    check_guess(
        r"
pub use foo::bar::baz;
#[doc(hidden)]
pub use foo::bar::qux;
",
        ImportGranularityGuess::Unknown,
    );
}

#[test]
fn guess_one_differing_attrs() {
    check_guess(
        r"
pub use {foo::bar::baz};
#[doc(hidden)]
pub use {foo::bar::qux};
",
        ImportGranularityGuess::One,
    );
}

#[test]
fn guess_skips_multiple_one_style_same_attrs() {
    check_guess(
        r"
#[doc(hidden)]
use {foo::bar::baz};
#[doc(hidden)]
use {foo::bar::qux};
",
        ImportGranularityGuess::Unknown,
    );
}

#[test]
fn guess_grouping_matters() {
    check_guess(
        r"
use foo::bar::baz;
use oof::bar::baz;
use foo::bar::qux;
",
        ImportGranularityGuess::Unknown,
    );
}

#[test]
fn insert_with_renamed_import_simple_use() {
    check_with_config(
        "use self::foo::Foo",
        r#"
use self::foo::Foo as _;
"#,
        r#"
use self::foo::Foo;
"#,
        &InsertUseConfig {
            granularity: ImportGranularity::Crate,
            prefix_kind: hir::PrefixKind::BySelf,
            enforce_granularity: true,
            group: true,
            skip_glob_imports: true,
        },
    );
}

#[test]
fn insert_with_renamed_import_complex_use() {
    check_with_config(
        "use self::foo::Foo;",
        r#"
use self::foo::{self, Foo as _, Bar};
"#,
        r#"
use self::foo::{self, Bar, Foo};
"#,
        &InsertUseConfig {
            granularity: ImportGranularity::Crate,
            prefix_kind: hir::PrefixKind::BySelf,
            enforce_granularity: true,
            group: true,
            skip_glob_imports: true,
        },
    );
}

#[test]
fn insert_with_double_colon_prefixed_import_merge() {
    check_with_config(
        "use ::ext::foo::Foo",
        r#"
use ::ext::foo::Foo as _;
"#,
        r#"
use ::ext::foo::Foo;
"#,
        &InsertUseConfig {
            granularity: ImportGranularity::Crate,
            prefix_kind: hir::PrefixKind::BySelf,
            enforce_granularity: true,
            group: true,
            skip_glob_imports: true,
        },
    );
}

fn check_with_config(
    path: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
    config: &InsertUseConfig,
) {
    let (db, file_id, pos) = if ra_fixture_before.contains(CURSOR_MARKER) {
        let (db, file_id, range_or_offset) = RootDatabase::with_range_or_offset(ra_fixture_before);

        (db, file_id, Some(range_or_offset))
    } else {
        let (db, file_id) = RootDatabase::with_single_file(ra_fixture_before);

        (db, file_id, None)
    };
    let sema = &Semantics::new(&db);
    let source_file = sema.parse(file_id);
    let file = pos
        .and_then(|pos| source_file.syntax().token_at_offset(pos.expect_offset()).next()?.parent())
        .and_then(|it| ImportScope::find_insert_use_container(&it, sema))
        .unwrap_or_else(|| ImportScope {
            kind: ImportScopeKind::File(source_file),
            required_cfgs: vec![],
        })
        .clone_for_update();
    let path = ast::SourceFile::parse(&format!("use {path};"), span::Edition::CURRENT)
        .tree()
        .syntax()
        .descendants()
        .find_map(ast::Path::cast)
        .unwrap();

    insert_use(&file, path, config);
    let result = file.as_syntax_node().ancestors().last().unwrap().to_string();
    assert_eq_text!(&trim_indent(ra_fixture_after), &result);
}

fn check(
    path: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
    granularity: ImportGranularity,
) {
    check_with_config(
        path,
        ra_fixture_before,
        ra_fixture_after,
        &InsertUseConfig {
            granularity,
            enforce_granularity: true,
            prefix_kind: PrefixKind::Plain,
            group: true,
            skip_glob_imports: true,
        },
    )
}

fn check_crate(
    path: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
) {
    check(path, ra_fixture_before, ra_fixture_after, ImportGranularity::Crate)
}

fn check_module(
    path: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
) {
    check(path, ra_fixture_before, ra_fixture_after, ImportGranularity::Module)
}

fn check_none(
    path: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
) {
    check(path, ra_fixture_before, ra_fixture_after, ImportGranularity::Item)
}

fn check_one(
    path: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
) {
    check(path, ra_fixture_before, ra_fixture_after, ImportGranularity::One)
}

fn check_merge_only_fail(ra_fixture0: &str, ra_fixture1: &str, mb: MergeBehavior) {
    let use0 = ast::SourceFile::parse(ra_fixture0, span::Edition::CURRENT)
        .tree()
        .syntax()
        .descendants()
        .find_map(ast::Use::cast)
        .unwrap();

    let use1 = ast::SourceFile::parse(ra_fixture1, span::Edition::CURRENT)
        .tree()
        .syntax()
        .descendants()
        .find_map(ast::Use::cast)
        .unwrap();

    let result = try_merge_imports(&use0, &use1, mb);
    assert_eq!(result.map(|u| u.to_string()), None);
}

fn check_guess(#[rust_analyzer::rust_fixture] ra_fixture: &str, expected: ImportGranularityGuess) {
    let syntax = ast::SourceFile::parse(ra_fixture, span::Edition::CURRENT).tree();
    let file = ImportScope { kind: ImportScopeKind::File(syntax), required_cfgs: vec![] };
    assert_eq!(super::guess_granularity_from_scope(&file), expected);
}
