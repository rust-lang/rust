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
    // Note: this is really an inference failure.
    // The correct answer would be to make X
    // equal to @const int, but we are not (yet)
    // smart enough.
    r(@3); //! ERROR (values differ in mutability)

}
