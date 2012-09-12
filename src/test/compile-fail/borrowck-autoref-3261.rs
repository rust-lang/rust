use either::*;
enum X = Either<(uint,uint),fn()>;
impl &X {
    fn with(blk: fn(x: &Either<(uint,uint),fn()>)) {
        blk(&**self)
    }
}
fn main() {
    let mut x = X(Right(main));
    do (&mut x).with |opt| {  //~ ERROR illegal borrow
        match *opt {
            Right(f) => {
                x = X(Left((0,0)));
                f()
            },
            _ => fail
        }
    }
}