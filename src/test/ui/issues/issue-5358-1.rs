enum Either<T, U> { Left(T), Right(U) }
struct S(Either<usize, usize>);

fn main() {
    match S(Either::Left(5)) {
        Either::Right(_) => {}
        //~^ ERROR mismatched types
        //~| expected type `S`
        //~| found type `Either<_, _>`
        //~| expected struct `S`, found enum `Either`
        _ => {}
    }
}
