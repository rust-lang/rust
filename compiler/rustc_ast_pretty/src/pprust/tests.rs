use super::*;

use rustc_ast as ast;
use rustc_span::create_default_session_globals_then;
use rustc_span::symbol::Ident;
use thin_vec::ThinVec;

fn fun_to_string(
    decl: &ast::FnDecl,
    header: ast::FnHeader,
    name: Ident,
    generics: &ast::Generics,
) -> String {
    to_string(|s| {
        s.head("");
        s.print_fn(decl, header, Some(name), generics);
        s.end(); // Close the head box.
        s.end(); // Close the outer box.
    })
}

fn variant_to_string(var: &ast::Variant) -> String {
    to_string(|s| s.print_variant(var))
}

#[test]
fn test_fun_to_string() {
    create_default_session_globals_then(|| {
        let abba_ident = Ident::from_str("abba");

        let decl = ast::FnDecl {
            inputs: ThinVec::new(),
            output: ast::FnRetTy::Default(rustc_span::DUMMY_SP),
        };
        let generics = ast::Generics::default();
        assert_eq!(
            fun_to_string(&decl, ast::FnHeader::default(), abba_ident, &generics),
            "fn abba()"
        );
    })
}

#[test]
fn test_variant_to_string() {
    create_default_session_globals_then(|| {
        let ident = Ident::from_str("principal_skinner");

        let var = ast::Variant {
            ident,
            vis: ast::Visibility {
                span: rustc_span::DUMMY_SP,
                kind: ast::VisibilityKind::Inherited,
                tokens: None,
            },
            attrs: ast::AttrVec::new(),
            id: ast::DUMMY_NODE_ID,
            data: ast::VariantData::Unit(ast::DUMMY_NODE_ID),
            disr_expr: None,
            span: rustc_span::DUMMY_SP,
            is_placeholder: false,
        };

        let varstr = variant_to_string(&var);
        assert_eq!(varstr, "principal_skinner");
    })
}
