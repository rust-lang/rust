//! FIXME: write short doc here

use super::*;

// test param_list
// fn a() {}
// fn b(x: i32) {}
// fn c(x: i32, ) {}
// fn d(x: i32, y: ()) {}
pub(super) fn param_list_fn(p: &mut Parser) {
    list_(p, Flavor::Function)
}

// test param_list_opt_patterns
// fn foo<F: FnMut(&mut Foo<'a>)>(){}
pub(super) fn param_list_impl_fn(p: &mut Parser) {
    list_(p, Flavor::ImplFn)
}

pub(super) fn param_list_fn_ptr(p: &mut Parser) {
    list_(p, Flavor::FnPointer)
}

pub(super) fn param_list_closure(p: &mut Parser) {
    list_(p, Flavor::Closure)
}

#[derive(Debug, Clone, Copy)]
enum Flavor {
    Function, // Includes trait fn params; omitted param idents are not supported
    ImplFn,
    FnPointer,
    Closure,
}

fn list_(p: &mut Parser, flavor: Flavor) {
    use Flavor::*;

    let (bra, ket) = match flavor {
        Closure => (T![|], T![|]),
        Function | ImplFn | FnPointer => (T!['('], T![')']),
    };

    let m = p.start();
    p.bump(bra);

    if let Function = flavor {
        // test self_param_outer_attr
        // fn f(#[must_use] self) {}
        attributes::outer_attributes(p);
        opt_self_param(p);
    }

    while !p.at(EOF) && !p.at(ket) {
        // test param_outer_arg
        // fn f(#[attr1] pat: Type) {}
        attributes::outer_attributes(p);

        if let Function | FnPointer = flavor {
            if p.at(T![...]) {
                break;
            }
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
    if let Function | FnPointer = flavor {
        p.eat(T![...]);
    }
    p.expect(ket);
    m.complete(p, PARAM_LIST);
}

const VALUE_PARAMETER_FIRST: TokenSet = patterns::PATTERN_FIRST.union(types::TYPE_FIRST);

fn value_parameter(p: &mut Parser, flavor: Flavor) {
    let m = p.start();
    match flavor {
        // test trait_fn_placeholder_parameter
        // trait Foo {
        //     fn bar(_: u64, mut x: i32);
        // }

        // test trait_fn_patterns
        // trait T {
        //     fn f1((a, b): (usize, usize)) {}
        //     fn f2(S { a, b }: S) {}
        //     fn f3(NewType(a): NewType) {}
        //     fn f4(&&a: &&usize) {}
        // }

        // test fn_patterns
        // impl U {
        //     fn f1((a, b): (usize, usize)) {}
        //     fn f2(S { a, b }: S) {}
        //     fn f3(NewType(a): NewType) {}
        //     fn f4(&&a: &&usize) {}
        // }
        Flavor::Function => {
            patterns::pattern(p);
            types::ascription(p);
        }
        // test value_parameters_no_patterns
        // type F = Box<Fn(i32, &i32, &i32, ())>;
        Flavor::ImplFn => {
            types::type_(p);
        }
        // test fn_pointer_param_ident_path
        // type Foo = fn(Bar::Baz);
        // type Qux = fn(baz: Bar::Baz);
        Flavor::FnPointer => {
            if p.at(IDENT) && p.nth(1) == T![:] && !p.nth_at(1, T![::]) {
                patterns::pattern(p);
                types::ascription(p);
            } else {
                types::type_(p);
            }
        }
        Flavor::Closure => {
            patterns::pattern(p);
            if p.at(T![:]) && !p.at(T![::]) {
                types::ascription(p);
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
            p.bump_any();
        }
    }
    m.complete(p, SELF_PARAM);
    if !p.at(T![')']) {
        p.expect(T![,]);
    }
}
