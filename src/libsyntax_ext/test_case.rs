// http://rust-lang.org/COPYRIGHT.
//

// #[test_case] is used by custom test authors to mark tests
// When building for test, it needs to make the item public and gensym the name
// Otherwise, we'll omit the item. This behavior means that any item annotated
// with #[test_case] is never addressable.
//
// We mark item with an inert attribute "rustc_test_marker" which the test generation
// logic will pick up on.

use syntax::ext::base::*;
use syntax::ext::build::AstBuilder;
use syntax::ext::hygiene::{Mark, SyntaxContext};
use syntax::ast;
use syntax::source_map::respan;
use syntax::symbol::sym;
use syntax_pos::Span;
use syntax::source_map::{ExpnInfo, MacroAttribute};
use syntax::feature_gate;

pub fn expand(
    ecx: &mut ExtCtxt<'_>,
    attr_sp: Span,
    _meta_item: &ast::MetaItem,
    anno_item: Annotatable
) -> Vec<Annotatable> {
    if !ecx.ecfg.enable_custom_test_frameworks() {
        feature_gate::emit_feature_err(&ecx.parse_sess,
                                       sym::custom_test_frameworks,
                                       attr_sp,
                                       feature_gate::GateIssue::Language,
                                       feature_gate::EXPLAIN_CUSTOM_TEST_FRAMEWORKS);
    }

    if !ecx.ecfg.should_test { return vec![]; }

    let sp = {
        let mark = Mark::fresh(Mark::root());
        mark.set_expn_info(ExpnInfo::with_unstable(
            MacroAttribute(sym::test_case), attr_sp, ecx.parse_sess.edition,
            &[sym::test, sym::rustc_attrs],
        ));
        attr_sp.with_ctxt(SyntaxContext::empty().apply_mark(mark))
    };

    let mut item = anno_item.expect_item();

    item = item.map(|mut item| {
        item.vis = respan(item.vis.span, ast::VisibilityKind::Public);
        item.ident = item.ident.gensym();
        item.attrs.push(
            ecx.attribute(sp,
                ecx.meta_word(sp, sym::rustc_test_marker))
        );
        item
    });

    return vec![Annotatable::Item(item)]
}
