use syntax::{ast, attr};
use syntax::edition::Edition;
use syntax::ext::hygiene::AstPass;
use syntax::ext::base::Resolver;
use syntax::ptr::P;
use syntax::source_map::respan;
use syntax::symbol::{Ident, Symbol, kw, sym};
use syntax_pos::DUMMY_SP;

pub fn inject(
    mut krate: ast::Crate,
    resolver: &mut dyn Resolver,
    alt_std_name: Option<Symbol>,
    edition: Edition,
) -> (ast::Crate, Option<Symbol>) {
    let rust_2018 = edition >= Edition::Edition2018;

    // the first name in this list is the crate name of the crate with the prelude
    let names: &[Symbol] = if attr::contains_name(&krate.attrs, sym::no_core) {
        return (krate, None);
    } else if attr::contains_name(&krate.attrs, sym::no_std) {
        if attr::contains_name(&krate.attrs, sym::compiler_builtins) {
            &[sym::core]
        } else {
            &[sym::core, sym::compiler_builtins]
        }
    } else {
        &[sym::std]
    };

    let expn_id = resolver.expansion_for_ast_pass(
        DUMMY_SP,
        AstPass::StdImports,
        &[sym::prelude_import],
        None,
    );
    let span = DUMMY_SP.with_def_site_ctxt(expn_id);
    let call_site = DUMMY_SP.with_call_site_ctxt(expn_id);

    // .rev() to preserve ordering above in combination with insert(0, ...)
    for &orig_name_sym in names.iter().rev() {
        let (rename, orig_name) = if rust_2018 {
            (Ident::new(kw::Underscore, span), Some(orig_name_sym))
        } else {
            (Ident::new(orig_name_sym, call_site), None)
        };
        krate.module.items.insert(0, P(ast::Item {
            attrs: vec![attr::mk_attr_outer(
                attr::mk_word_item(ast::Ident::new(sym::macro_use, span))
            )],
            vis: respan(span, ast::VisibilityKind::Inherited),
            node: ast::ItemKind::ExternCrate(alt_std_name.or(orig_name)),
            ident: rename,
            id: ast::DUMMY_NODE_ID,
            span,
            tokens: None,
        }));
    }

    // the crates have been injected, the assumption is that the first one is the one with
    // the prelude.
    let name = names[0];

    let segments = if rust_2018 {
        [name, sym::prelude, sym::v1].iter()
            .map(|symbol| ast::PathSegment::from_ident(ast::Ident::new(*symbol, span)))
            .collect()
    } else {
        [kw::PathRoot, name, sym::prelude, sym::v1].iter()
            .map(|symbol| ast::PathSegment::from_ident(ast::Ident::new(*symbol, call_site)))
            .collect()
    };

    let use_item = P(ast::Item {
        attrs: vec![attr::mk_attr_outer(
            attr::mk_word_item(ast::Ident::new(sym::prelude_import, span)))],
        vis: respan(span.shrink_to_lo(), ast::VisibilityKind::Inherited),
        node: ast::ItemKind::Use(P(ast::UseTree {
            prefix: ast::Path { segments, span },
            kind: ast::UseTreeKind::Glob,
            span,
        })),
        id: ast::DUMMY_NODE_ID,
        ident: ast::Ident::invalid(),
        span,
        tokens: None,
    });

    let prelude_import_item = if rust_2018 {
        let hygienic_extern_crate = P(ast::Item {
            attrs: vec![],
            vis: respan(span, ast::VisibilityKind::Inherited),
            node: ast::ItemKind::ExternCrate(alt_std_name),
            ident: ast::Ident::new(name, span),
            id: ast::DUMMY_NODE_ID,
            span,
            tokens: None,
        });

        // Use an anonymous const to hide `extern crate std as hygienic_std`
        // FIXME: Once inter-crate hygiene exists, this can just be `use_item`.
        P(ast::Item {
            attrs: Vec::new(),
            vis: respan(span.shrink_to_lo(), ast::VisibilityKind::Inherited),
            node: ast::ItemKind::Const(
                P(ast::Ty {
                    id: ast::DUMMY_NODE_ID,
                    node: ast::TyKind::Tup(Vec::new()),
                    span,
                }),
                P(ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    attrs: syntax::ThinVec::new(),
                    node: ast::ExprKind::Block(P(ast::Block {
                        id: ast::DUMMY_NODE_ID,
                        rules: ast::BlockCheckMode::Default,
                        stmts: vec![
                            ast::Stmt {
                                id: ast::DUMMY_NODE_ID,
                                node: ast::StmtKind::Item(use_item),
                                span,
                            },
                            ast::Stmt {
                                id: ast::DUMMY_NODE_ID,
                                node: ast::StmtKind::Item(hygienic_extern_crate),
                                span,
                            }
                        ],
                        span,
                    }), None),
                    span,
                })
            ),
            id: ast::DUMMY_NODE_ID,
            ident: ast::Ident::new(kw::Underscore, span),
            span,
            tokens: None,
        })
    } else {
        // Have `extern crate std` at the root, so don't need to create a named
        // extern crate item.
        use_item
    };

    krate.module.items.insert(0, prelude_import_item);

    (krate, Some(name))
}
