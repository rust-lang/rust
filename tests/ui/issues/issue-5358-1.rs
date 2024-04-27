enum Either<T, U> { Left(T), Right(U) }
struct S(Either<usize, usize>);

fn main() {
    match S(Either::Left(5)) {
        Either::Right(_) => {}
        //~^ ERROR mismatched types
        //~| expected `S`, found `Either<_, _>`
        //~| expected struct `S`
        //~| found enum `Either<_, _>`
        _ => {}
    }
}
