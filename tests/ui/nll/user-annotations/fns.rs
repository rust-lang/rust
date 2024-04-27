// Unit test for the "user substitutions" that are annotated on each
// node.

fn some_fn<T>(arg: T) { }

fn no_annot() {
    let c = 66;
    some_fn(&c); // OK
}

fn annot_underscore() {
    let c = 66;
    some_fn::<_>(&c); // OK
}

fn annot_reference_any_lifetime() {
    let c = 66;
    some_fn::<&u32>(&c); // OK
}

fn annot_reference_static_lifetime() {
    let c = 66;
    some_fn::<&'static u32>(&c); //~ ERROR
}

fn annot_reference_named_lifetime<'a>(_d: &'a u32) {
    let c = 66;
    some_fn::<&'a u32>(&c); //~ ERROR
}

fn annot_reference_named_lifetime_ok<'a>(c: &'a u32) {
    some_fn::<&'a u32>(c);
}

fn annot_reference_named_lifetime_in_closure<'a>(_: &'a u32) {
    let _closure = || {
        let c = 66;
        some_fn::<&'a u32>(&c); //~ ERROR
    };
}

fn annot_reference_named_lifetime_in_closure_ok<'a>(c: &'a u32) {
    let _closure = || {
        some_fn::<&'a u32>(c);
    };
}

fn main() { }
