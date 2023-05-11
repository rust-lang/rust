use crate::grammar::attributes::ATTRIBUTE_FIRST;

use super::*;

// test param_list
// fn a() {}
// fn b(x: i32) {}
// fn c(x: i32, ) {}
// fn d(x: i32, y: ()) {}
pub(super) fn param_list_fn_def(p: &mut Parser<'_>) {
    list_(p, Flavor::FnDef);
}

// test param_list_opt_patterns
// fn foo<F: FnMut(&mut Foo<'a>)>(){}
pub(super) fn param_list_fn_trait(p: &mut Parser<'_>) {
    list_(p, Flavor::FnTrait);
}

pub(super) fn param_list_fn_ptr(p: &mut Parser<'_>) {
    list_(p, Flavor::FnPointer);
}

pub(super) fn param_list_closure(p: &mut Parser<'_>) {
    list_(p, Flavor::Closure);
}

#[derive(Debug, Clone, Copy)]
enum Flavor {
    FnDef,   // Includes trait fn params; omitted param idents are not supported
    FnTrait, // Params for `Fn(...)`/`FnMut(...)`/`FnOnce(...)` annotations
    FnPointer,
    Closure,
}

fn list_(p: &mut Parser<'_>, flavor: Flavor) {
    use Flavor::*;

    let (bra, ket) = match flavor {
        Closure => (T![|], T![|]),
        FnDef | FnTrait | FnPointer => (T!['('], T![')']),
    };

    let list_marker = p.start();
    p.bump(bra);

    let mut param_marker = None;
    if let FnDef = flavor {
        // test self_param_outer_attr
        // fn f(#[must_use] self) {}
        let m = p.start();
        attributes::outer_attrs(p);
        match opt_self_param(p, m) {
            Ok(()) => {}
            Err(m) => param_marker = Some(m),
        }
    }

    while !p.at(EOF) && !p.at(ket) {
        // test param_outer_arg
        // fn f(#[attr1] pat: Type) {}
        let m = match param_marker.take() {
            Some(m) => m,
            None => {
                let m = p.start();
                attributes::outer_attrs(p);
                m
            }
        };

        if !p.at_ts(PARAM_FIRST.union(ATTRIBUTE_FIRST)) {
            p.error("expected value parameter");
            m.abandon(p);
            break;
        }
        param(p, m, flavor);
        if !p.at(T![,]) {
            if p.at_ts(PARAM_FIRST.union(ATTRIBUTE_FIRST)) {
                p.error("expected `,`");
            } else {
                break;
            }
        } else {
            p.bump(T![,]);
        }
    }

    if let Some(m) = param_marker {
        m.abandon(p);
    }

    p.expect(ket);
    list_marker.complete(p, PARAM_LIST);
}

const PARAM_FIRST: TokenSet = patterns::PATTERN_FIRST.union(types::TYPE_FIRST);

fn param(p: &mut Parser<'_>, m: Marker, flavor: Flavor) {
    match flavor {
        // test param_list_vararg
        // extern "C" { fn printf(format: *const i8, ..., _: u8) -> i32; }
        Flavor::FnDef | Flavor::FnPointer if p.eat(T![...]) => {}

        // test fn_def_param
        // fn foo(..., (x, y): (i32, i32)) {}
        Flavor::FnDef => {
            patterns::pattern(p);
            if !variadic_param(p) {
                if p.at(T![:]) {
                    types::ascription(p);
                } else {
                    // test_err missing_fn_param_type
                    // fn f(x y: i32, z, t: i32) {}
                    p.error("missing type for function parameter");
                }
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
                if !variadic_param(p) {
                    if p.at(T![:]) {
                        types::ascription(p);
                    } else {
                        p.error("missing type for function parameter");
                    }
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
}

fn variadic_param(p: &mut Parser<'_>) -> bool {
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
fn opt_self_param(p: &mut Parser<'_>, m: Marker) -> Result<(), Marker> {
    if p.at(T![self]) || p.at(T![mut]) && p.nth(1) == T![self] {
        p.eat(T![mut]);
        self_as_name(p);
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
        if !matches!(
            (p.current(), la1, la2, la3),
            (T![&], T![self], _, _)
                | (T![&], T![mut] | LIFETIME_IDENT, T![self], _)
                | (T![&], LIFETIME_IDENT, T![mut], T![self])
        ) {
            return Err(m);
        }
        p.bump(T![&]);
        if p.at(LIFETIME_IDENT) {
            lifetime(p);
        }
        p.eat(T![mut]);
        self_as_name(p);
    }
    m.complete(p, SELF_PARAM);
    if !p.at(T![')']) {
        p.expect(T![,]);
    }
    Ok(())
}

fn self_as_name(p: &mut Parser<'_>) {
    let m = p.start();
    p.bump(T![self]);
    m.complete(p, NAME);
}
