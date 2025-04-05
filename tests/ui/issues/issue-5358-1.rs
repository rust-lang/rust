enum Either<T, U> { Left(T), Right(U) }
struct S(Either<usize, usize>);

fn main() {
    match S(Either::Left(5)) {
        Either::Right(_) => {}
        //~^ ERROR mismatched types
        //~| NOTE_NONVIRAL expected `S`, found `Either<_, _>`
        //~| NOTE_NONVIRAL expected struct `S`
        //~| NOTE_NONVIRAL found enum `Either<_, _>`
        _ => {}
    }
}
