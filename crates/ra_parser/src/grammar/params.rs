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
    let (bra, ket) = if flavor.type_required() { (T!['('], T![')']) } else { (T![|], T![|]) };
    assert!(p.at(bra));
    let m = p.start();
    p.bump();
    if flavor.type_required() {
        // test self_param_outer_attr
        // fn f(#[must_use] self) {}
        attributes::outer_attributes(p);
        opt_self_param(p);
    }
    while !p.at(EOF) && !p.at(ket) {
        // test param_outer_arg
        // fn f(#[attr1] pat: Type) {}
        attributes::outer_attributes(p);

        if flavor.type_required() && p.at(T![...]) {
            break;
        }

        if !p.at_ts(VALUE_PARAMETER_FIRST) {
            p.error("expected value parameter");
            break;
        }
        value_parameter(p, flavor);
        if !p.at(ket) {
            p.expect(T![,]);
        }
    }
    // test param_list_vararg
    // extern "C" { fn printf(format: *const i8, ...) -> i32; }
    if flavor.type_required() {
        p.eat(T![...]);
    }
    p.expect(ket);
    m.complete(p, PARAM_LIST);
}

const VALUE_PARAMETER_FIRST: TokenSet = patterns::PATTERN_FIRST.union(types::TYPE_FIRST);

fn value_parameter(p: &mut Parser, flavor: Flavor) {
    let m = p.start();
    match flavor {
        Flavor::OptionalType | Flavor::Normal => {
            patterns::pattern(p);
            if p.at(T![:]) || flavor.type_required() {
                types::ascription(p)
            }
        }
        // test value_parameters_no_patterns
        // type F = Box<Fn(a: i32, &b: &i32, &mut c: &i32, ())>;
        Flavor::OptionalPattern => {
            let la0 = p.current();
            let la1 = p.nth(1);
            let la2 = p.nth(2);
            let la3 = p.nth(3);

            // test trait_fn_placeholder_parameter
            // trait Foo {
            //     fn bar(_: u64, mut x: i32);
            // }
            if (la0 == IDENT || la0 == T![_]) && la1 == T![:]
                || la0 == T![mut] && la1 == IDENT && la2 == T![:]
                || la0 == T![&] && la1 == IDENT && la2 == T![:]
                || la0 == T![&] && la1 == T![mut] && la2 == IDENT && la3 == T![:]
            {
                patterns::pattern(p);
                types::ascription(p);
            } else {
                types::type_(p);
            }
        }
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
    if p.at(T![self]) || p.at(T![mut]) && p.nth(1) == T![self] {
        m = p.start();
        p.eat(T![mut]);
        p.eat(T![self]);
        // test arb_self_types
        // impl S {
        //     fn a(self: &Self) {}
        //     fn b(mut self: Box<Self>) {}
        // }
        if p.at(T![:]) {
            types::ascription(p);
        }
    } else {
        let la1 = p.nth(1);
        let la2 = p.nth(2);
        let la3 = p.nth(3);
        let n_toks = match (p.current(), la1, la2, la3) {
            (T![&], T![self], _, _) => 2,
            (T![&], T![mut], T![self], _) => 3,
            (T![&], LIFETIME, T![self], _) => 3,
            (T![&], LIFETIME, T![mut], T![self]) => 4,
            _ => return,
        };
        m = p.start();
        for _ in 0..n_toks {
            p.bump();
        }
    }
    m.complete(p, SELF_PARAM);
    if !p.at(T![')']) {
        p.expect(T![,]);
    }
}
