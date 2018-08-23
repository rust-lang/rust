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
        opt_self_param(p);
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
    match flavor {
        Flavor::OptionalType | Flavor::Normal => {
            patterns::pattern(p);
            if p.at(COLON) || flavor.type_required() {
                types::ascription(p)
            }
        },
        // test value_parameters_no_patterns
        // type F = Box<Fn(a: i32, &b: &i32, &mut c: &i32, ())>;
        Flavor::OptionalPattern => {
            let la0 = p.current();
            let la1 = p.nth(1);
            let la2 = p.nth(2);
            let la3 = p.nth(3);
            if la0 == IDENT && la1 == COLON
                || la0 == AMP && la1 == IDENT && la2 == COLON
                || la0 == AMP && la1 == MUT_KW && la2 == IDENT && la3 == COLON {
                patterns::pattern(p);
                types::ascription(p);
            } else {
                types::type_(p);
            }
        },
    }
    m.complete(p, PARAM);
}

// test self_param
// impl S {
//     fn a(self) {}
//     fn b(&self,) {}
//     fn c(&'a self,) {}
//     fn d(&'a mut self, x: i32) {}
//     fn e(mut self) {}
// }
fn opt_self_param(p: &mut Parser) {
    let m;
    if p.at(SELF_KW) || p.at(MUT_KW) && p.nth(1) == SELF_KW {
        m = p.start();
        p.eat(MUT_KW);
        p.eat(SELF_KW);
        // test arb_self_types
        // impl S {
        //     fn a(self: &Self) {}
        //     fn b(mut self: Box<Self>) {}
        // }
        if p.at(COLON) {
            types::ascription(p);
        }
    } else {
        let la1 = p.nth(1);
        let la2 = p.nth(2);
        let la3 = p.nth(3);
        let n_toks = match (p.current(), la1, la2, la3) {
            (AMP, SELF_KW, _, _) => 2,
            (AMP, MUT_KW, SELF_KW, _) => 3,
            (AMP, LIFETIME, SELF_KW, _) => 3,
            (AMP, LIFETIME, MUT_KW, SELF_KW) => 4,
            _ => return,
        };
        m = p.start();
        for _ in 0..n_toks {
            p.bump();
        }
    }
    m.complete(p, SELF_PARAM);
    if !p.at(R_PAREN) {
        p.expect(COMMA);
    }
}
