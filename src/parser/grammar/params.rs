use super::*;

// test param_list
// fn a() {}
// fn b(x: i32) {}
// fn c(x: i32, ) {}
// fn d(x: i32, y: ()) {}
pub(super) fn list(p: &mut Parser) {
    list_(p, true)
}

pub(super) fn list_opt_types(p: &mut Parser) {
    list_(p, false)
}

fn list_(p: &mut Parser, require_types: bool) {
    assert!(p.at(if require_types { L_PAREN } else { PIPE }));
    let m = p.start();
    p.bump();
    if require_types {
        self_param(p);
    }
    let terminator = if require_types { R_PAREN } else { PIPE };
    while !p.at(EOF) && !p.at(terminator) {
        value_parameter(p, require_types);
        if !p.at(terminator) {
            p.expect(COMMA);
        }
    }
    p.expect(terminator);
    m.complete(p, PARAM_LIST);
}

fn value_parameter(p: &mut Parser, require_type: bool) {
    let m = p.start();
    patterns::pattern(p);
    if p.at(COLON) || require_type {
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
        (AMPERSAND, SELF_KW, _, _) => 2,
        (AMPERSAND, MUT_KW, SELF_KW, _) => 3,
        (AMPERSAND, LIFETIME, SELF_KW, _) => 3,
        (AMPERSAND, LIFETIME, MUT_KW, SELF_KW) => 4,
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

