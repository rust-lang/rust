use m::S;

mod m {
    pub struct S(u8);

    mod n {
        use S;
        fn f() {
            S(10);
            //~^ ERROR expected function, tuple struct or tuple variant, found struct `S`
        }
    }
}

fn main() {}
