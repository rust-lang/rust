// Unit test for the "user substitutions" that are annotated on each
// node.

#![allow(warnings)]

use std::cell::Cell;

enum SomeEnum<T> {
    SomeVariant(T),
    SomeOtherVariant,
}

fn combine<T>(_: T, _: T) { }

fn no_annot() {
    let c = 66;
    combine(SomeEnum::SomeVariant(Cell::new(&c)), SomeEnum::SomeOtherVariant);
}

fn annot_underscore() {
    let c = 66;
    combine(SomeEnum::SomeVariant(Cell::new(&c)), SomeEnum::SomeOtherVariant::<Cell<_>>);
}

fn annot_reference_any_lifetime() {
    let c = 66;
    combine(SomeEnum::SomeVariant(Cell::new(&c)), SomeEnum::SomeOtherVariant::<Cell<&u32>>);
}

fn annot_reference_static_lifetime() {
    let c = 66;
    combine(
        SomeEnum::SomeVariant(Cell::new(&c)), //~ ERROR
        SomeEnum::SomeOtherVariant::<Cell<&'static u32>>,
    );
}

fn annot_reference_named_lifetime<'a>(_d: &'a u32) {
    let c = 66;
    combine(
        SomeEnum::SomeVariant(Cell::new(&c)), //~ ERROR
        SomeEnum::SomeOtherVariant::<Cell<&'a u32>>,
    );
}

fn annot_reference_named_lifetime_ok<'a>(c: &'a u32) {
    combine(SomeEnum::SomeVariant(Cell::new(c)), SomeEnum::SomeOtherVariant::<Cell<&'a u32>>);
}

fn annot_reference_named_lifetime_in_closure<'a>(_: &'a u32) {
    let _closure = || {
        let c = 66;
        combine(
            SomeEnum::SomeVariant(Cell::new(&c)), //~ ERROR
            SomeEnum::SomeOtherVariant::<Cell<&'a u32>>,
        );
    };
}

fn annot_reference_named_lifetime_in_closure_ok<'a>(c: &'a u32) {
    let _closure = || {
        combine(
            SomeEnum::SomeVariant(Cell::new(c)),
            SomeEnum::SomeOtherVariant::<Cell<&'a u32>>,
        );
    };
}

fn main() { }
