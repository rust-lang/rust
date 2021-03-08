use super::*;

use hir::PrefixKind;
use test_utils::assert_eq_text;

#[test]
fn insert_not_group() {
    check(
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
        None,
        false,
        false,
    );
}

#[test]
fn insert_existing() {
    check_full("std::fs", "use std::fs;", "use std::fs;")
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
    cov_mark::check!(insert_use_indent_after);
    check_none(
        "std::bar::AA",
        r"
    use std::bar::B;
    use std::bar::D;",
        r"
    use std::bar::AA;
    use std::bar::B;
    use std::bar::D;",
    )
}

#[test]
fn insert_middle() {
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
    )
}

#[test]
fn insert_end() {
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
    cov_mark::check!(insert_use_indent_before);
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
    )
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
    check_full(
        "foo::bar",
        "fn main() {}",
        r"use foo::bar;

fn main() {}",
    )
}

#[test]
fn insert_empty_file() {
    // empty files will get two trailing newlines
    // this is due to the test case insert_no_imports above
    check_full(
        "foo::bar",
        "",
        r"use foo::bar;

",
    )
}

#[test]
fn insert_empty_module() {
    cov_mark::check!(insert_use_no_indent_after);
    check(
        "foo::bar",
        "mod x {}",
        r"{
    use foo::bar;
}",
        None,
        true,
        true,
    )
}

#[test]
fn insert_after_inner_attr() {
    check_full(
        "foo::bar",
        r"#![allow(unused_imports)]",
        r"#![allow(unused_imports)]

use foo::bar;",
    )
}

#[test]
fn insert_after_inner_attr2() {
    check_full(
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
    check_last("std::io", r"use std::fmt;", r"use std::{fmt, io};")
}

#[test]
fn merge_groups_last() {
    check_last(
        "std::io",
        r"use std::fmt::{Result, Display};",
        r"use std::fmt::{Result, Display};
use std::io;",
    )
}

#[test]
fn merge_last_into_self() {
    check_last("foo::bar::baz", r"use foo::bar;", r"use foo::bar::{self, baz};");
}

#[test]
fn merge_groups_full() {
    check_full(
        "std::io",
        r"use std::fmt::{Result, Display};",
        r"use std::{fmt::{Result, Display}, io};",
    )
}

#[test]
fn merge_groups_long_full() {
    check_full("std::foo::bar::Baz", r"use std::foo::bar::Qux;", r"use std::foo::bar::{Baz, Qux};")
}

#[test]
fn merge_groups_long_last() {
    check_last("std::foo::bar::Baz", r"use std::foo::bar::Qux;", r"use std::foo::bar::{Baz, Qux};")
}

#[test]
fn merge_groups_long_full_list() {
    check_full(
        "std::foo::bar::Baz",
        r"use std::foo::bar::{Qux, Quux};",
        r"use std::foo::bar::{Baz, Quux, Qux};",
    )
}

#[test]
fn merge_groups_long_last_list() {
    check_last(
        "std::foo::bar::Baz",
        r"use std::foo::bar::{Qux, Quux};",
        r"use std::foo::bar::{Baz, Quux, Qux};",
    )
}

#[test]
fn merge_groups_long_full_nested() {
    check_full(
        "std::foo::bar::Baz",
        r"use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
        r"use std::foo::bar::{Baz, Qux, quux::{Fez, Fizz}};",
    )
}

#[test]
fn merge_groups_long_last_nested() {
    check_last(
        "std::foo::bar::Baz",
        r"use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
        r"use std::foo::bar::Baz;
use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
    )
}

#[test]
fn merge_groups_full_nested_deep() {
    check_full(
        "std::foo::bar::quux::Baz",
        r"use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
        r"use std::foo::bar::{Qux, quux::{Baz, Fez, Fizz}};",
    )
}

#[test]
fn merge_groups_full_nested_long() {
    check_full(
        "std::foo::bar::Baz",
        r"use std::{foo::bar::Qux};",
        r"use std::{foo::bar::{Baz, Qux}};",
    );
}

#[test]
fn merge_groups_last_nested_long() {
    check_full(
        "std::foo::bar::Baz",
        r"use std::{foo::bar::Qux};",
        r"use std::{foo::bar::{Baz, Qux}};",
    );
}

#[test]
fn merge_groups_skip_pub() {
    check_full(
        "std::io",
        r"pub use std::fmt::{Result, Display};",
        r"pub use std::fmt::{Result, Display};
use std::io;",
    )
}

#[test]
fn merge_groups_skip_pub_crate() {
    check_full(
        "std::io",
        r"pub(crate) use std::fmt::{Result, Display};",
        r"pub(crate) use std::fmt::{Result, Display};
use std::io;",
    )
}

#[test]
fn merge_groups_skip_attributed() {
    check_full(
        "std::io",
        r#"
#[cfg(feature = "gated")] use std::fmt::{Result, Display};
"#,
        r#"
#[cfg(feature = "gated")] use std::fmt::{Result, Display};
use std::io;
"#,
    )
}

#[test]
#[ignore] // FIXME: Support this
fn split_out_merge() {
    check_last(
        "std::fmt::Result",
        r"use std::{fmt, io};",
        r"use std::fmt::{self, Result};
use std::io;",
    )
}

#[test]
fn merge_into_module_import() {
    check_full("std::fmt::Result", r"use std::{fmt, io};", r"use std::{fmt::{self, Result}, io};")
}

#[test]
fn merge_groups_self() {
    check_full("std::fmt::Debug", r"use std::fmt;", r"use std::fmt::{self, Debug};")
}

#[test]
fn merge_mod_into_glob() {
    check_full("token::TokenKind", r"use token::TokenKind::*;", r"use token::TokenKind::{*, self};")
    // FIXME: have it emit `use token::TokenKind::{self, *}`?
}

#[test]
fn merge_self_glob() {
    check_full("self", r"use self::*;", r"use self::{*, self};")
    // FIXME: have it emit `use {self, *}`?
}

#[test]
fn merge_glob_nested() {
    check_full(
        "foo::bar::quux::Fez",
        r"use foo::bar::{Baz, quux::*};",
        r"use foo::bar::{Baz, quux::{self::*, Fez}};",
    )
}

#[test]
fn merge_nested_considers_first_segments() {
    check_full(
        "hir_ty::display::write_bounds_like_dyn_trait",
        r"use hir_ty::{autoderef, display::{HirDisplayError, HirFormatter}, method_resolution};",
        r"use hir_ty::{autoderef, display::{HirDisplayError, HirFormatter, write_bounds_like_dyn_trait}, method_resolution};",
    );
}

#[test]
fn skip_merge_last_too_long() {
    check_last(
        "foo::bar",
        r"use foo::bar::baz::Qux;",
        r"use foo::bar;
use foo::bar::baz::Qux;",
    );
}

#[test]
fn skip_merge_last_too_long2() {
    check_last(
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
        MergeBehavior::Last,
    );
}

#[test]
fn merge_last_fail1() {
    check_merge_only_fail(
        r"use foo::bar::{baz::{Qux, Fez}};",
        r"use foo::bar::baaz::{Quux, Feez};",
        MergeBehavior::Last,
    );
}

#[test]
fn merge_last_fail2() {
    check_merge_only_fail(
        r"use foo::bar::baz::{Qux, Fez};",
        r"use foo::bar::{baaz::{Quux, Feez}};",
        MergeBehavior::Last,
    );
}

#[test]
fn merge_last_fail3() {
    check_merge_only_fail(
        r"use foo::bar::baz::{Qux, Fez};",
        r"use foo::bar::baaz::{Quux, Feez};",
        MergeBehavior::Last,
    );
}

fn check(
    path: &str,
    ra_fixture_before: &str,
    ra_fixture_after: &str,
    mb: Option<MergeBehavior>,
    module: bool,
    group: bool,
) {
    let mut syntax = ast::SourceFile::parse(ra_fixture_before).tree().syntax().clone();
    if module {
        syntax = syntax.descendants().find_map(ast::Module::cast).unwrap().syntax().clone();
    }
    let file = super::ImportScope::from(syntax).unwrap();
    let path = ast::SourceFile::parse(&format!("use {};", path))
        .tree()
        .syntax()
        .descendants()
        .find_map(ast::Path::cast)
        .unwrap();

    let rewriter = insert_use(
        &file,
        path,
        InsertUseConfig { merge: mb, prefix_kind: PrefixKind::Plain, group },
    );
    let result = rewriter.rewrite(file.as_syntax_node()).to_string();
    assert_eq_text!(ra_fixture_after, &result);
}

fn check_full(path: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
    check(path, ra_fixture_before, ra_fixture_after, Some(MergeBehavior::Full), false, true)
}

fn check_last(path: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
    check(path, ra_fixture_before, ra_fixture_after, Some(MergeBehavior::Last), false, true)
}

fn check_none(path: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
    check(path, ra_fixture_before, ra_fixture_after, None, false, true)
}

fn check_merge_only_fail(ra_fixture0: &str, ra_fixture1: &str, mb: MergeBehavior) {
    let use0 = ast::SourceFile::parse(ra_fixture0)
        .tree()
        .syntax()
        .descendants()
        .find_map(ast::Use::cast)
        .unwrap();

    let use1 = ast::SourceFile::parse(ra_fixture1)
        .tree()
        .syntax()
        .descendants()
        .find_map(ast::Use::cast)
        .unwrap();

    let result = try_merge_imports(&use0, &use1, mb);
    assert_eq!(result.map(|u| u.to_string()), None);
}
