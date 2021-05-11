// Regression test for issue #83538. The problem here is that we have
// two cycles:
//
// * `Ty` embeds `Box<Ty>` indirectly, which depends on `Global: 'static`, which is OkModuloRegions.
// * But `Ty` also references `First`, which has a cycle on itself. That should just be `Ok`.
//
// But our caching mechanism was blending both cycles and giving the incorrect result.

#![feature(rustc_attrs)]
#![allow(bad_style)]

struct First {
    b: Vec<First>,
}

pub struct Second {
    d: Vec<First>,
}

struct Third<f> {
    g: Vec<f>,
}

enum Ty {
    j(Fourth, Fifth, Sixth),
}

struct Fourth {
    o: Vec<Ty>,
}

struct Fifth {
    bounds: First,
}

struct Sixth {
    p: Box<Ty>,
}

#[rustc_evaluate_where_clauses]
fn forward()
where
    Vec<First>: Unpin,
    Third<Ty>: Unpin,
{
}

#[rustc_evaluate_where_clauses]
fn reverse()
where
    Third<Ty>: Unpin,
    Vec<First>: Unpin,
{
}

fn main() {
    // Key is that Vec<First> is "ok" and Third<Ty> is "ok modulo regions":

    forward();
    //~^ ERROR evaluate(Binder(TraitPredicate(<std::vec::Vec<First> as std::marker::Unpin>), [])) = Ok(EvaluatedToOk)
    //~| ERROR evaluate(Binder(TraitPredicate(<Third<Ty> as std::marker::Unpin>), [])) = Ok(EvaluatedToOkModuloRegions)

    reverse();
    //~^ ERROR evaluate(Binder(TraitPredicate(<std::vec::Vec<First> as std::marker::Unpin>), [])) = Ok(EvaluatedToOk)
    //~| ERROR evaluate(Binder(TraitPredicate(<Third<Ty> as std::marker::Unpin>), [])) = Ok(EvaluatedToOkModuloRegions)
}
