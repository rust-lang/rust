// Unit test for the "user substitutions" that are annotated on each
// node.

trait Bazoom<T> {
    fn method<U>(&self, arg: T, arg2: U) { }
}

impl<T, U> Bazoom<U> for T {
}

fn no_annot() {
    let a = 22;
    let b = 44;
    let c = 66;
    a.method(b,  &c); // OK
}

fn annot_underscore() {
    let a = 22;
    let b = 44;
    let c = 66;
    a.method::<_>(b,  &c); // OK
}

fn annot_reference_any_lifetime() {
    let a = 22;
    let b = 44;
    let c = 66;
    a.method::<&u32>(b,  &c); // OK
}

fn annot_reference_static_lifetime() {
    let a = 22;
    let b = 44;
    let c = 66;
    a.method::<&'static u32>(b,  &c); //~ ERROR
}

fn annot_reference_named_lifetime<'a>(_d: &'a u32) {
    let a = 22;
    let b = 44;
    let c = 66;
    a.method::<&'a u32>(b,  &c); //~ ERROR
}

fn annot_reference_named_lifetime_ok<'a>(c: &'a u32) {
    let a = 22;
    let b = 44;
    a.method::<&'a u32>(b,  c);
}

fn annot_reference_named_lifetime_in_closure<'a>(_: &'a u32) {
    let a = 22;
    let b = 44;
    let _closure = || {
        let c = 66;
        a.method::<&'a u32>(b,  &c); //~ ERROR
    };
}

fn annot_reference_named_lifetime_in_closure_ok<'a>(c: &'a u32) {
    let a = 22;
    let b = 44;
    let _closure = || {
        a.method::<&'a u32>(b,  c);
    };
}

fn main() { }
