// run-pass
fn tuple() {
    struct S;
    struct Z;
    struct W;
    let x = (S, Z, W);
    match x { (S, ..) => {} }
    match x { (.., W) => {} }
    match x { (S, .., W) => {} }
    match x { (.., Z, _) => {} }
}

fn tuple_struct() {
    struct SS(S, Z, W);

    struct S;
    struct Z;
    struct W;
    let x = SS(S, Z, W);
    match x { SS(S, ..) => {} }
    match x { SS(.., W) => {} }
    match x { SS(S, .., W) => {} }
    match x { SS(.., Z, _) => {} }
}

fn main() {
    tuple();
    tuple_struct();
}
