use m::S;

mod m {
    pub struct S(u8);

    mod n {
        use crate::S;
        fn f() {
            S(10);
            //~^ ERROR cannot find function, tuple struct or tuple variant `S` in this scope
        }
    }
}

fn main() {}
