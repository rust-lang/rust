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
    <_ as Bazoom<_>>::method(&a, b, &c); // OK
}

fn annot_underscore() {
    let a = 22;
    let b = 44;
    let c = 66;
    <_ as Bazoom<_>>::method::<_>(&a, b, &c); // OK
}

fn annot_reference_any_lifetime() {
    let a = 22;
    let b = 44;
    let c = 66;
    <_ as Bazoom<_>>::method::<&u32>(&a, b, &c); // OK
}

fn annot_reference_static_lifetime() {
    let a = 22;
    let b = 44;
    let c = 66;
    <_ as Bazoom<_>>::method::<&'static u32>(&a, b, &c); //~ ERROR
}

fn annot_reference_named_lifetime<'a>(_d: &'a u32) {
    let a = 22;
    let b = 44;
    let c = 66;
    <_ as Bazoom<_>>::method::<&'a u32>(&a, b, &c); //~ ERROR
}

fn annot_reference_named_lifetime_ok<'a>(c: &'a u32) {
    let a = 22;
    let b = 44;
    <_ as Bazoom<_>>::method::<&'a u32>(&a, b, c);
}

fn annot_reference_named_lifetime_in_closure<'a>(_: &'a u32) {
    let a = 22;
    let b = 44;
    let _closure = || {
        let c = 66;
        <_ as Bazoom<_>>::method::<&'a u32>(&a, b, &c); //~ ERROR
    };
}

fn annot_reference_named_lifetime_in_closure_ok<'a>(c: &'a u32) {
    let a = 22;
    let b = 44;
    let _closure = || {
        <_ as Bazoom<_>>::method::<&'a u32>(&a, b, c);
    };
}

fn main() { }
