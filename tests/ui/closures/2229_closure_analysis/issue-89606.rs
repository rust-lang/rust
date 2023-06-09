// Regression test for #89606. Used to ICE.
//
// check-pass
// revisions: twenty_eighteen twenty_twentyone
// [twenty_eighteen]compile-flags: --edition 2018
// [twenty_twentyone]compile-flags: --edition 2021

struct S<'a>(Option<&'a mut i32>);

fn by_ref(s: &mut S<'_>) {
    (|| {
        let S(_o) = s;
        s.0 = None;
    })();
}

fn by_value(s: S<'_>) {
    (|| {
        let S(ref _o) = s;
        let _g = s.0;
    })();
}

struct V<'a>((Option<&'a mut i32>,));

fn nested(v: &mut V<'_>) {
    (|| {
        let V((_o,)) = v;
        v.0 = (None, );
    })();
}

fn main() {
    let mut s = S(None);
    by_ref(&mut s);
    by_value(s);

    let mut v = V((None, ));
    nested(&mut v);
}
