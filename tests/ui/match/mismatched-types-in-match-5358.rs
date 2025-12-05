// https://github.com/rust-lang/rust/issues/5358
enum Either<T, U> {
    Left(T),
    Right(U),
}
struct S(Either<usize, usize>);

fn main() {
    match S(Either::Left(5)) {
        //~^ NOTE this expression has type `S`
        Either::Right(_) => {}
        //~^ ERROR mismatched types
        //~| NOTE expected `S`, found `Either<_, _>`
        //~| NOTE expected struct `S`
        //~| NOTE found enum `Either<_, _>`
        _ => {}
    }
}
