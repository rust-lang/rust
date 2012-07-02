fn mk_identity<T:copy>() -> fn@(T) -> T {
    fn@(t: T) -> T { t }
}

fn main() {
    // type of r is fn@(X) -> X
    // for some fresh X
    let r = mk_identity();

    // @mut int <: X
    r(@mut 3);

    // @int <: X
    //
    // This constraint forces X to be
    // @const int.
    r(@3);

    // Here the type check succeeds but the
    // mutability check will fail, because the
    // type of r has been inferred to be
    // fn(@const int) -> @const int
    *r(@mut 3) = 4; //~ ERROR assigning to dereference of const @ pointer
}
