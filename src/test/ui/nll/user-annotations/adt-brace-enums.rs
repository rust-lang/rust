// Unit test for the "user substitutions" that are annotated on each
// node.

enum SomeEnum<T> {
    SomeVariant { t: T }
}

fn no_annot() {
    let c = 66;
    SomeEnum::SomeVariant { t: &c };
}

fn annot_underscore() {
    let c = 66;
    SomeEnum::<_>::SomeVariant { t: &c };
}

fn annot_underscore2() {
    let c = 66;
    SomeEnum::SomeVariant::<_> { t: &c }; //~ WARNING
}

fn annot_reference_any_lifetime() {
    let c = 66;
    SomeEnum::<&u32>::SomeVariant { t: &c };
}

fn annot_reference_any_lifetime2() {
    let c = 66;
    SomeEnum::SomeVariant::<&u32> { t: &c }; //~ WARNING
}

fn annot_reference_static_lifetime() {
    let c = 66;
    SomeEnum::<&'static u32>::SomeVariant { t: &c }; //~ ERROR
}

fn annot_reference_static_lifetime2() {
    let c = 66;
    SomeEnum::SomeVariant::<&'static u32> { t: &c }; //~ ERROR
    //~^ WARNING
}

fn annot_reference_named_lifetime<'a>(_d: &'a u32) {
    let c = 66;
    SomeEnum::<&'a u32>::SomeVariant { t: &c }; //~ ERROR
}

fn annot_reference_named_lifetime2<'a>(_d: &'a u32) {
    let c = 66;
    SomeEnum::SomeVariant::<&'a u32> { t: &c }; //~ ERROR
    //~^ WARNING
}

fn annot_reference_named_lifetime_ok<'a>(c: &'a u32) {
    SomeEnum::<&'a u32>::SomeVariant { t: c };
}

fn annot_reference_named_lifetime_ok2<'a>(c: &'a u32) {
    SomeEnum::SomeVariant::<&'a u32> { t: c }; //~ WARNING
}

fn annot_reference_named_lifetime_in_closure<'a>(_: &'a u32) {
    let _closure = || {
        let c = 66;
        SomeEnum::<&'a u32>::SomeVariant { t: &c }; //~ ERROR
    };
}

fn annot_reference_named_lifetime_in_closure2<'a>(_: &'a u32) {
    let _closure = || {
        let c = 66;
        SomeEnum::SomeVariant::<&'a u32> { t: &c }; //~ ERROR
        //~^ WARNING
    };
}

fn annot_reference_named_lifetime_in_closure_ok<'a>(c: &'a u32) {
    let _closure = || {
        SomeEnum::<&'a u32>::SomeVariant { t: c };
    };
}

fn annot_reference_named_lifetime_in_closure_ok2<'a>(c: &'a u32) {
    let _closure = || {
        SomeEnum::SomeVariant::<&'a u32> { t: c }; //~ WARNING
    };
}

fn main() { }
