//@ check-pass
use std::error::Error;
struct A {
}

impl A {
    pub fn new() -> A {
        A {
        }
    }

    pub fn f<'a>(
        &'a self,
        team_name: &'a str,
        c: &'a mut dyn FnMut(String, String, u64, u64)
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}


fn main() {
    let a = A::new();
    let participant_name = "A";

    let c = |a, b, c, d| {};

    a.f(participant_name, &mut c); //~ WARNING cannot borrow
}
