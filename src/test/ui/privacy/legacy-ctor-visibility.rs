use m::S;

mod m {
    pub struct S(u8);

    mod n {
        use S;
        fn f() {
            S(10);
            //~^ ERROR expected function, found struct `S`
        }
    }
}

fn main() {}
