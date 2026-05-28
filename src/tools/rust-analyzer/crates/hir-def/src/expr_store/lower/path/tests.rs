use expect_test::{Expect, expect};
use span::Edition;
use syntax::ast::{self, make};
use test_fixture::WithFixture;

use crate::{
    expr_store::{
        ExpressionStore,
        lower::{
            ExprCollector,
            path::{SEGMENT_LOWERING_MAP, hir_segment_to_ast_segment},
        },
        path::Path,
        pretty,
    },
    nameres::crate_def_map,
    test_db::TestDB,
};

fn lower_path(path: ast::Path) -> (TestDB, ExpressionStore, Option<Path>) {
    let (db, file_id) = TestDB::with_single_file("");
    let krate = db.fetch_test_crate();
    let mut ctx =
        ExprCollector::new(&db, crate_def_map(&db, krate).root_module_id(), file_id.into());
    let lowered_path = ctx.lower_path(path, &mut ExprCollector::impl_trait_allocator);
    let (store, _) = ctx.store.finish();
    (db, store, lowered_path)
}

#[track_caller]
fn check_hir_to_ast(path: &str, ignore_segments: &[&str]) {
    let path = make::path_from_text(path);
    SEGMENT_LOWERING_MAP.with_borrow_mut(|map| map.clear());
    let _ = lower_path(path.clone()).2.expect("failed to lower path");
    SEGMENT_LOWERING_MAP.with_borrow(|map| {
        for (segment, segment_idx) in map {
            if ignore_segments.contains(&&*segment.to_string()) {
                continue;
            }

            let restored_segment = hir_segment_to_ast_segment(&path, *segment_idx as u32)
                .unwrap_or_else(|| {
                    panic!(
                        "failed to map back segment `{segment}` \
                        numbered {segment_idx} in HIR from path `{path}`"
                    )
                });
            assert_eq!(
                segment, &restored_segment,
                "mapping back `{segment}` numbered {segment_idx} in HIR \
                from path `{path}` produced incorrect segment `{restored_segment}`"
            );
        }
    });
}

#[test]
fn hir_to_ast_trait_ref() {
    check_hir_to_ast("<A as B::C::D>::E::F", &["A"]);
}

#[test]
fn hir_to_ast_plain_path() {
    check_hir_to_ast("A::B::C::D::E::F", &[]);
}

#[test]
fn hir_to_ast_crate_path() {
    check_hir_to_ast("crate::A::B::C", &[]);
    check_hir_to_ast("crate::super::super::A::B::C", &[]);
}

#[test]
fn hir_to_ast_self_path() {
    check_hir_to_ast("self::A::B::C", &[]);
    check_hir_to_ast("self::super::super::A::B::C", &[]);
}

#[test]
fn hir_to_ast_super_path() {
    check_hir_to_ast("super::A::B::C", &[]);
    check_hir_to_ast("super::super::super::A::B::C", &[]);
}

#[test]
fn hir_to_ast_type_anchor_path() {
    check_hir_to_ast("<A::B>::C::D", &["A", "B"]);
}

#[test]
fn hir_to_ast_path_super_in_middle() {
    check_hir_to_ast("A::super::B::super::super::C::D", &[]);
}

#[track_caller]
fn check_fail_lowering(path: &str) {
    let (_, _, lowered_path) = lower_path(make::path_from_text(path));
    assert!(lowered_path.is_none(), "path `{path}` should fail lowering");
}

#[test]
fn keywords_in_middle_fail_lowering1() {
    check_fail_lowering("self::A::self::B::super::C::crate::D");
}

#[test]
fn keywords_in_middle_fail_lowering2() {
    check_fail_lowering("A::super::self::C::D");
}

#[test]
fn keywords_in_middle_fail_lowering3() {
    check_fail_lowering("A::crate::B::C::D");
}

#[track_caller]
fn check_path_lowering(path: &str, expected: Expect) {
    let (db, store, lowered_path) = lower_path(make::path_from_text(path));
    let lowered_path = lowered_path.expect("failed to lower path");
    let buf = pretty::print_path(&db, &store, &lowered_path, Edition::CURRENT);
    expected.assert_eq(&buf);
}

#[test]
fn fn_like_path_with_coloncolon() {
    check_path_lowering("Fn::(A, B) -> C", expect![[r#"Fn::<(A, B), Output = C>"#]]);
    check_path_lowering("Fn::(A, B)", expect![[r#"Fn::<(A, B), Output = ()>"#]]);
}
