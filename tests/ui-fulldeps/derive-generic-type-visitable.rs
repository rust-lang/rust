//@ edition: 2024
//@ run-pass

#![feature(rustc_private)]

extern crate rustc_type_ir;
extern crate rustc_type_ir_macros;

use rustc_type_ir::GenericTypeVisitable;
use rustc_type_ir_macros::GenericTypeVisitable;

// Necessary to pull in object code as the rest of the rustc crates are shipped only as rmeta
// files.
#[expect(unused_extern_crates)]
extern crate rustc_driver;

#[derive(GenericTypeVisitable)]
struct DerivesGenericTypeVisitable;

#[derive(GenericTypeVisitable)]
struct Foo {
    one: Incrementer,
    two: Vec<Incrementer>,
}

#[derive(GenericTypeVisitable)]
enum Enum {
    A,
    B(Incrementer),
    C { one: Incrementer, two: Vec<Incrementer> },
}

#[derive(GenericTypeVisitable)]
struct Generic<T>(Vec<T>);

#[derive(GenericTypeVisitable)]
struct Recursive {
    #[generic_type_visitable(bounds())]
    rec: Vec<Self>,
    other: Incrementer,
}

#[derive(GenericTypeVisitable)]
struct PartiallyRecursiveField<T> {
    #[generic_type_visitable(bounds(T: GenericTypeVisitable<__V>))]
    partially_rec: (Vec<Self>, T),
    other: Incrementer,
}

// start testing setup

use std::sync::atomic::{AtomicU8, Ordering};

static COUNT: AtomicU8 = AtomicU8::new(0);

/// A type that, when visited, increments a global counter.
///
/// Used to (weakly) test the correctness of the derive by making sure that
/// it traverses all the fields, and thus reaches all the incrementers.
#[derive(Clone)]
struct Incrementer;

unsafe impl<V> GenericTypeVisitable<V> for Incrementer {
    fn generic_visit_with(&self, _visitor: &mut V) {
        COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

// end testing setup

fn main() {
    use Incrementer as Inc; // for brevity

    #[track_caller]
    fn check<T: GenericTypeVisitable<()>>(item: T, count: u8) {
        let mut v = ();
        item.generic_visit_with(&mut v);
        assert_eq!(COUNT.swap(0, Ordering::Relaxed), count);
    }

    check(DerivesGenericTypeVisitable, 0);
    check(Foo { one: Inc, two: vec![] }, 1);
    check(Foo { one: Inc, two: vec![Inc; 2] }, 1 + 2);
    check(Enum::A, 0);
    check(Enum::B(Inc), 1);
    check(Enum::C { one: Inc, two: vec![] }, 1);
    check(Enum::C { one: Inc, two: vec![Inc; 3] }, 1 + 3);
    check(Generic::<Inc>(vec![]), 0);
    // visits each of the nested `Inc`s
    check(Generic(vec![Inc; 5]), 5);

    // Every (nested) `rec!` adds another `Recursive`, and thus 1 more visited `Inc`.
    macro_rules! rec {
        [$($i:expr),* $(,)?] => {
            Recursive { rec: vec![$($i),*], other: Inc }
        }
    }
    check(rec![], 1);
    check(rec![rec![]], 2);
    check(rec![rec![], rec![]], 3);
    check(rec![rec![rec![]]], 3);

    // Every (nested) `prec!` adds another `PartiallyRecursiveField`, and thus 1 more visited `Inc`.
    macro_rules! prec {
        ([$($i:expr),* $(,)?], $o:expr) => {
            PartiallyRecursiveField { partially_rec: (vec![$($i),*], $o), other: Inc }
        }
    }
    // Every nested `a()`, `b()`, and `c()` adds 0, 1, and 2 more visited `Inc`s, respectively.
    let a = || Enum::A;
    let b = || Enum::B(Inc);
    let c = || Enum::C { one: Inc, two: vec![Inc] };
    check(prec!([], a()), 1 + 0);
    check(prec!([], b()), 1 + 1);
    check(prec!([], c()), 1 + 2);
    check(prec!([prec!([], a())], a()), 1 + (1 + 0) + 0);
    check(prec!([prec!([], b())], a()), 1 + (1 + 1) + 0);
    check(prec!([prec!([], b())], b()), 1 + (1 + 1) + 1);
}
