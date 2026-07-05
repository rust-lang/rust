// Regression test for the uniqueness guard added alongside the #146126 fix.
//
// The E0282 retargeting heuristic walks from a stuck inference variable, through
// `Subtype`/`Coerce` relations, to a stalled associated type projection, and
// blames that projection's unresolved input. When *several* distinct projections
// end up in the same inference "component" (here two different `.get(_.into())`
// results joined by an `if`/`else`), there is no single obvious culprit, so the
// heuristic must bail out and leave the ordinary "type annotations needed"
// diagnostic in place rather than arbitrarily blaming one of them.

struct Foo {
    f: Option<i32>,
}

fn main() {
    let a = vec![Foo { f: Some(1) }];
    let b = vec![Foo { f: Some(2) }];
    let x = if true { a.get(0u8.into()) } else { b.get(0u8.into()) };
    //~^ ERROR type annotations needed
    if let Some(y) = x {
        if let Some(_f) = y.f {}
    }
}
