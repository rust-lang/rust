//@ check-pass
#![feature(late_bound_turbofishing)]

fn foo_early<'a: 'a>(b: &'a u32) -> &'a u32 { b }
fn foo_late<'a>(b: &'a u32) -> &'a u32 { b }

trait Trait {
    type Assoc<'a>;
}

// zero explicit generic lifetimes
fn do_thing<T: Trait>(_: Option<<T as Trait>::Assoc<'_>>) -> &i32 {
    todo!()
}

// one explicit generic lifetime
fn do_thing_2<'b, T: Trait>(_: Option<<T as Trait>::Assoc<'_>>) -> (&i32, &'b i64) {
    todo!()
}

// zero explicit generic lifetimes
fn do_thing_3<T: Trait>(_: Option<<T as Trait>::Assoc<'_>>) -> i32 {
    todo!()
}

fn foo<T: Trait>() {
    // one explicit generic lifetime
    do_thing::<'static, T>(None);
    // two explicit generic lifetimes
    do_thing_2::<'static, 'static, T>(None);
    // one explicit generic lifetime
    do_thing_3::<'static, T>(None);
}

fn require_static<T: 'static>(_: T) { }

fn main() {
    let f = foo_early::<'static>;
    require_static(f);
    let f = foo_early::<'static>;
    require_static(f);
}
