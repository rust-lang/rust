// run-pass

// Test that move closures compile properly with `capture_disjoint_fields` enabled.

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete

fn simple_ref() {
    let mut s = 10;
    let ref_s = &mut s;

    let mut c = move || {
        *ref_s += 10;
    };
    c();
}

fn struct_contains_ref_to_another_struct() {
    struct S(String);
    struct T<'a>(&'a mut S);

    let mut s = S("s".into());
    let t = T(&mut s);

    let mut c = move || {
        t.0.0 = "new s".into();
    };

    c();
}

fn no_ref() {
    struct S(String);
    struct T(S);

    let t = T(S("s".into()));
    let mut c = move || {
        t.0.0 = "new S".into();
    };
    c();
}

fn main() {
    simple_ref();
    struct_contains_ref_to_another_struct();
    no_ref();
}
