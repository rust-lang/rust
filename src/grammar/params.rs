use super::*;

// test param_list
// fn a() {}
// fn b(x: i32) {}
// fn c(x: i32, ) {}
// fn d(x: i32, y: ()) {}
pub(super) fn param_list(p: &mut Parser) {
    list_(p, Flavor::Normal)
}

// test param_list_opt_patterns
// fn foo<F: FnMut(&mut Foo<'a>)>(){}
pub(super) fn param_list_opt_patterns(p: &mut Parser) {
    list_(p, Flavor::OptionalPattern)
}

pub(super) fn param_list_opt_types(p: &mut Parser) {
    list_(p, Flavor::OptionalType)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Flavor {
    OptionalType,
    OptionalPattern,
    Normal,
}

impl Flavor {
    fn type_required(self) -> bool {
        match self {
            Flavor::OptionalType => false,
            _ => true,
        }
    }
    fn pattern_required(self) -> bool {
        match self {
            Flavor::OptionalPattern => false,
            _ => true,
        }
    }
}

fn list_(p: &mut Parser, flavor: Flavor) {
    let (bra, ket) = if flavor.type_required() {
        (L_PAREN, R_PAREN)
    } else {
        (PIPE, PIPE)
    };
    assert!(p.at(bra));
    let m = p.start();
    p.bump();
    if flavor.type_required() {
        self_param(p);
    }
    while !p.at(EOF) && !p.at(ket) {
        value_parameter(p, flavor);
        if !p.at(ket) {
            p.expect(COMMA);
        }
    }
    p.expect(ket);
    m.complete(p, PARAM_LIST);
}

fn value_parameter(p: &mut Parser, flavor: Flavor) {
    let m = p.start();
    patterns::pattern(p);
    if p.at(COLON) || flavor.type_required() {
        types::ascription(p)
    }
    m.complete(p, PARAM);
}

// test self_param
// impl S {
//     fn a(self) {}
//     fn b(&self,) {}
//     fn c(&'a self,) {}
//     fn d(&'a mut self, x: i32) {}
// }
fn self_param(p: &mut Parser) {
    let la1 = p.nth(1);
    let la2 = p.nth(2);
    let la3 = p.nth(3);
    let n_toks = match (p.current(), la1, la2, la3) {
        (SELF_KW, _, _, _) => 1,
        (AMP, SELF_KW, _, _) => 2,
        (AMP, MUT_KW, SELF_KW, _) => 3,
        (AMP, LIFETIME, SELF_KW, _) => 3,
        (AMP, LIFETIME, MUT_KW, SELF_KW) => 4,
        _ => return,
    };
    let m = p.start();
    for _ in 0..n_toks {
        p.bump();
    }
    m.complete(p, SELF_PARAM);
    if !p.at(R_PAREN) {
        p.expect(COMMA);
    }
}
