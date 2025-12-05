// Unit test for the "user substitutions" that are annotated on each
// node.

struct SomeStruct<T>(T);

fn no_annot() {
    let c = 66;
    let f = SomeStruct;
    f(&c);
}

fn annot_underscore() {
    let c = 66;
    let f = SomeStruct::<_>;
    f(&c);
}

fn annot_reference_any_lifetime() {
    let c = 66;
    let f = SomeStruct::<&u32>;
    f(&c);
}

fn annot_reference_static_lifetime() {
    let c = 66;
    let f = SomeStruct::<&'static u32>;
    f(&c); //~ ERROR
}

fn annot_reference_named_lifetime<'a>(_d: &'a u32) {
    let c = 66;
    let f = SomeStruct::<&'a u32>;
    f(&c); //~ ERROR
}

fn annot_reference_named_lifetime_ok<'a>(c: &'a u32) {
    let f = SomeStruct::<&'a u32>;
    f(c);
}

fn annot_reference_named_lifetime_in_closure<'a>(_: &'a u32) {
    let _closure = || {
        let c = 66;
        let f = SomeStruct::<&'a u32>;
        f(&c); //~ ERROR
    };
}

fn annot_reference_named_lifetime_across_closure<'a>(_: &'a u32) {
    let f = SomeStruct::<&'a u32>;
    let _closure = || {
        let c = 66;
        f(&c); //~ ERROR
    };
}

fn annot_reference_named_lifetime_in_closure_ok<'a>(c: &'a u32) {
    let _closure = || {
        let f = SomeStruct::<&'a u32>;
        f(c);
    };
}

fn annot_reference_named_lifetime_across_closure_ok<'a>(c: &'a u32) {
    let f = SomeStruct::<&'a u32>;
    let _closure = || {
        f(c);
    };
}

fn main() { }
