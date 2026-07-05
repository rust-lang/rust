//@ revisions: current next
//@[next] compile-flags: -Znext-solver
// Regression test for #146126.
//
// When the target type of a `.into()` passed as an argument to a generic method
// (here `<[T]>::get`, whose return type is the associated type projection
// `<I as SliceIndex<[T]>>::Output`) cannot be inferred, the ambiguity used to be
// reported against a downstream use of the call's result (the `x.f` field
// access) instead of the `.into()` that is its actual source.
//
// Check that the error now points inside `v.get(0_i16.into())`, not at `x.f`.
// The `next` revision exercises the `-Znext-solver` code paths (`NormalizesTo` /
// `AliasRelate` obligations) of the retargeting heuristic.

struct Foo {
    f: Option<i32>,
}

fn main() {
    let v = vec![Foo { f: Some(1) }];
    if let Some(x) = v.get(0_i16.into()) {
        //~^ ERROR type annotations needed
        if let Some(f) = x.f {
            println!("{}", f);
        } else {
            println!("not present");
        }
    } else {
        println!("bad index");
    }
}
