use rustc_ast as ast;
use rustc_ast::mut_visit::MutVisitor;
use rustc_ast_pretty::pprust;
use rustc_span::{Ident, create_default_session_globals_then};

use crate::parser::tests::{matches_codepattern, string_to_crate};

// This version doesn't care about getting comments or doc-strings in.
fn print_crate_items(krate: &ast::Crate) -> String {
    krate.items.iter().map(|i| pprust::item_to_string(i)).collect::<Vec<_>>().join(" ")
}

// Change every identifier to "zz".
struct ToZzIdentMutVisitor;

impl MutVisitor for ToZzIdentMutVisitor {
    const VISIT_TOKENS: bool = true;

    fn visit_ident(&mut self, ident: &mut Ident) {
        *ident = Ident::from_str("zz");
    }
}

macro_rules! assert_matches_codepattern {
    ($a:expr , $b:expr) => {{
        let a_val = $a;
        let b_val = $b;
        if !matches_codepattern(&a_val, &b_val) {
            panic!("expected args satisfying `matches_codepattern`, got {} and {}", a_val, b_val);
        }
    }};
}

// Make sure idents get transformed everywhere.
#[test]
fn ident_transformation() {
    create_default_session_globals_then(|| {
        let mut zz_visitor = ToZzIdentMutVisitor;
        let mut krate =
            string_to_crate("#[a] mod b {fn c (d : e, f : g) {h!(i,j,k);l;m}}".to_string());
        zz_visitor.visit_crate(&mut krate);
        assert_matches_codepattern!(
            print_crate_items(&krate),
            "#[zz]mod zz{fn zz(zz:zz,zz:zz){zz!(zz,zz,zz);zz;zz}}".to_string()
        );
    })
}

// Make sure idents get transformed even inside macro defs.
#[test]
fn ident_transformation_in_defs() {
    create_default_session_globals_then(|| {
        let mut zz_visitor = ToZzIdentMutVisitor;
        let mut krate = string_to_crate(
            "macro_rules! a {(b $c:expr $(d $e:token)f+ => \
            (g $(d $d $e)+))} "
                .to_string(),
        );
        zz_visitor.visit_crate(&mut krate);
        assert_matches_codepattern!(
            print_crate_items(&krate),
            "macro_rules! zz{(zz$zz:zz$(zz $zz:zz)zz+=>(zz$(zz$zz$zz)+))}".to_string()
        );
    })
}
