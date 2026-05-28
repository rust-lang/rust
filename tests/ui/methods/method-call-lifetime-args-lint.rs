#![deny(late_bound_lifetime_arguments)]
#![allow(unused)]

struct S;

impl S {
    fn late<'a, 'b>(self, _: &'a u8, _: &'b u8) {}
    fn late_implicit(self, _: &u8, _: &u8) {}
}

fn method_call() {
    S.late::<'static>(&0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted

    S.late_implicit::<'static>(&0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
}

fn main() {}
