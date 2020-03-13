//! FIXME: write short doc here

use super::*;

// test param_list
// fn a() {}
// fn b(x: i32) {}
// fn c(x: i32, ) {}
// fn d(x: i32, y: ()) {}
pub(super) fn param_list_fn_def(p: &mut Parser) {
    list_(p, Flavor::FnDef)
}

// test param_list_opt_patterns
// fn foo<F: FnMut(&mut Foo<'a>)>(){}
pub(super) fn param_list_fn_trait(p: &mut Parser) {
    list_(p, Flavor::FnTrait)
}

pub(super) fn param_list_fn_ptr(p: &mut Parser) {
    list_(p, Flavor::FnPointer)
}

pub(super) fn param_list_closure(p: &mut Parser) {
    list_(p, Flavor::Closure)
}

#[derive(Debug, Clone, Copy)]
enum Flavor {
    FnDef,   // Includes trait fn params; omitted param idents are not supported
    FnTrait, // Params for `Fn(...)`/`FnMut(...)`/`FnOnce(...)` annotations
    FnPointer,
    Closure,
}

fn list_(p: &mut Parser, flavor: Flavor) {
    use Flavor::*;

    let (bra, ket) = match flavor {
        Closure => (T![|], T![|]),
        FnDef | FnTrait | FnPointer => (T!['('], T![')']),
    };

    let m = p.start();
    p.bump(bra);

    if let FnDef = flavor {
        // test self_param_outer_attr
        // fn f(#[must_use] self) {}
        attributes::outer_attributes(p);
        opt_self_param(p);
    }

    while !p.at(EOF) && !p.at(ket) {
        // test param_outer_arg
        // fn f(#[attr1] pat: Type) {}
        attributes::outer_attributes(p);

        if !p.at_ts(VALUE_PARAMETER_FIRST) {
            p.error("expected value parameter");
            break;
        }
        let param = value_parameter(p, flavor);
        if !p.at(ket) {
            p.expect(T![,]);
        }
        if let Variadic(true) = param {
            break;
        }
    }

    p.expect(ket);
    m.complete(p, PARAM_LIST);
}

const VALUE_PARAMETER_FIRST: TokenSet = patterns::PATTERN_FIRST.union(types::TYPE_FIRST);

struct Variadic(bool);

fn value_parameter(p: &mut Parser, flavor: Flavor) -> Variadic {
    let mut res = Variadic(false);
    let m = p.start();
    match flavor {
        // test param_list_vararg
        // extern "C" { fn printf(format: *const i8, ...) -> i32; }
        Flavor::FnDef | Flavor::FnPointer if p.eat(T![...]) => res = Variadic(true),

        // test fn_def_param
        // fn foo((x, y): (i32, i32)) {}
        Flavor::FnDef => {
            patterns::pattern(p);
            if variadic_param(p) {
                res = Variadic(true)
            } else {
                types::ascription(p);
            }
        }
        // test value_parameters_no_patterns
        // type F = Box<Fn(i32, &i32, &i32, ())>;
        Flavor::FnTrait => {
            types::type_(p);
        }
        // test fn_pointer_param_ident_path
        // type Foo = fn(Bar::Baz);
        // type Qux = fn(baz: Bar::Baz);

        // test fn_pointer_unnamed_arg
        // type Foo = fn(_: bar);
        Flavor::FnPointer => {
            if (p.at(IDENT) || p.at(UNDERSCORE)) && p.nth(1) == T![:] && !p.nth_at(1, T![::]) {
                patterns::pattern_single(p);
                if variadic_param(p) {
                    res = Variadic(true)
                } else {
                    types::ascription(p);
                }
            } else {
                types::type_(p);
            }
        }
        // test closure_params
        // fn main() {
        //    let foo = |bar, baz: Baz, qux: Qux::Quux| ();
        // }
        Flavor::Closure => {
            patterns::pattern_single(p);
            if p.at(T![:]) && !p.at(T![::]) {
                types::ascription(p);
            }
        }
    }
    m.complete(p, PARAM);
    res
}

fn variadic_param(p: &mut Parser) -> bool {
    if p.at(T![:]) && p.nth_at(1, T![...]) {
        p.bump(T![:]);
        p.bump(T![...]);
        true
    } else {
        false
    }
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
            p.bump_any();
        }
    }
    m.complete(p, SELF_PARAM);
    if !p.at(T![')']) {
        p.expect(T![,]);
    }
}
