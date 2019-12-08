use crate::tests::{string_to_crate, matches_codepattern};

use syntax::ast::{self, Ident};
use syntax::print::pprust;
use syntax::mut_visit::{self, MutVisitor};
use syntax::with_default_globals;

// This version doesn't care about getting comments or doc-strings in.
fn fake_print_crate(s: &mut pprust::State<'_>,
                    krate: &ast::Crate) {
    s.print_mod(&krate.module, &krate.attrs)
}

// Change every identifier to "zz".
struct ToZzIdentMutVisitor;

impl MutVisitor for ToZzIdentMutVisitor {
    fn visit_ident(&mut self, ident: &mut ast::Ident) {
        *ident = Ident::from_str("zz");
    }
    fn visit_mac(&mut self, mac: &mut ast::Mac) {
        mut_visit::noop_visit_mac(mac, self)
    }
}

// Maybe add to `expand.rs`.
macro_rules! assert_pred {
    ($pred:expr, $predname:expr, $a:expr , $b:expr) => (
        {
            let pred_val = $pred;
            let a_val = $a;
            let b_val = $b;
            if !(pred_val(&a_val, &b_val)) {
                panic!("expected args satisfying {}, got {} and {}",
                        $predname, a_val, b_val);
            }
        }
    )
}

// Make sure idents get transformed everywhere.
#[test] fn ident_transformation () {
    with_default_globals(|| {
        let mut zz_visitor = ToZzIdentMutVisitor;
        let mut krate = string_to_crate(
            "#[a] mod b {fn c (d : e, f : g) {h!(i,j,k);l;m}}".to_string());
        zz_visitor.visit_crate(&mut krate);
        assert_pred!(
            matches_codepattern,
            "matches_codepattern",
            pprust::to_string(|s| fake_print_crate(s, &krate)),
            "#[zz]mod zz{fn zz(zz:zz,zz:zz){zz!(zz,zz,zz);zz;zz}}".to_string());
    })
}

// Make sure idents get transformed even inside macro defs.
#[test] fn ident_transformation_in_defs () {
    with_default_globals(|| {
        let mut zz_visitor = ToZzIdentMutVisitor;
        let mut krate = string_to_crate(
            "macro_rules! a {(b $c:expr $(d $e:token)f+ => \
            (g $(d $d $e)+))} ".to_string());
        zz_visitor.visit_crate(&mut krate);
        assert_pred!(
            matches_codepattern,
            "matches_codepattern",
            pprust::to_string(|s| fake_print_crate(s, &krate)),
            "macro_rules! zz{(zz$zz:zz$(zz $zz:zz)zz+=>(zz$(zz$zz$zz)+))}".to_string());
    })
}
