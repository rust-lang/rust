struct S;

impl S {
    fn late<'a, 'b>(self, _: &'a u8, _: &'b u8) {}
    fn late_implicit(self, _: &u8, _: &u8) {}
}

fn ufcs() {
    S::late::<'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_implicit::<'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
}

fn main() {}
