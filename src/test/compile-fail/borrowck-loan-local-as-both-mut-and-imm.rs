use core::either::{Either, Left, Right};

    fn f(x: &mut Either<int,float>, y: &Either<int,float>) -> int {
        match *y {
            Left(ref z) => {
                *x = Right(1.0);
                *z
            }
            _ => fail
        }
    }

    fn g() {
        let mut x: Either<int,float> = Left(3);
        io::println(f(&mut x, &x).to_str()); //~ ERROR conflicts with prior loan
    }

    fn h() {
        let mut x: Either<int,float> = Left(3);
        let y: &Either<int, float> = &x;
        let z: &mut Either<int, float> = &mut x; //~ ERROR conflicts with prior loan
        *z = *y;
    } 

    fn main() {}
