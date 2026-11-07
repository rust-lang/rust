// Unit test for the "user substitutions" that are annotated on each
// node.

struct SomeStruct<T>(T);

fn no_annot() {
    let c = 66;
    SomeStruct(&c);
}

fn annot_underscore() {
    let c = 66;
    SomeStruct::<_>(&c);
}

fn annot_reference_any_lifetime() {
    let c = 66;
    SomeStruct::<&u32>(&c);
}

fn annot_reference_static_lifetime() {
    let c = 66;
    SomeStruct::<&'static u32>(&c); //~ ERROR
}

fn annot_reference_named_lifetime<'a>(_d: &'a u32) {
    let c = 66;
    SomeStruct::<&'a u32>(&c); //~ ERROR
}

fn annot_reference_named_lifetime_ok<'a>(c: &'a u32) {
    SomeStruct::<&'a u32>(c);
}

fn annot_reference_named_lifetime_in_closure<'a>(_: &'a u32) {
    let _closure = || {
        let c = 66;
        SomeStruct::<&'a u32>(&c); //~ ERROR
    };
}

fn annot_reference_named_lifetime_in_closure_ok<'a>(c: &'a u32) {
    let _closure = || {
        SomeStruct::<&'a u32>(c);
    };
}

fn main() { }
