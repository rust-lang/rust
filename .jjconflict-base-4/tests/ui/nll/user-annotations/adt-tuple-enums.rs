// Unit test for the "user substitutions" that are annotated on each
// node.

#![allow(warnings)]

enum SomeEnum<T> {
    SomeVariant(T),
    SomeOtherVariant,
}

fn no_annot() {
    let c = 66;
    SomeEnum::SomeVariant(&c);
}

fn annot_underscore() {
    let c = 66;
    SomeEnum::SomeVariant::<_>(&c);
}

fn annot_reference_any_lifetime() {
    let c = 66;
    SomeEnum::SomeVariant::<&u32>(&c);
}

fn annot_reference_static_lifetime() {
    let c = 66;
    SomeEnum::SomeVariant::<&'static u32>(&c); //~ ERROR
}

fn annot_reference_named_lifetime<'a>(_d: &'a u32) {
    let c = 66;
    SomeEnum::SomeVariant::<&'a u32>(&c); //~ ERROR
}

fn annot_reference_named_lifetime_ok<'a>(c: &'a u32) {
    SomeEnum::SomeVariant::<&'a u32>(c);
}

fn annot_reference_named_lifetime_in_closure<'a>(_: &'a u32) {
    let _closure = || {
        let c = 66;
        SomeEnum::SomeVariant::<&'a u32>(&c); //~ ERROR
    };
}

fn annot_reference_named_lifetime_in_closure_ok<'a>(c: &'a u32) {
    let _closure = || {
        SomeEnum::SomeVariant::<&'a u32>(c);
    };
}

fn main() { }
