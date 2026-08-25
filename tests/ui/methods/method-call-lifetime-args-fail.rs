struct S;

impl S {
    fn late<'a, 'b>(self, _: &'a u8, _: &'b u8) {}
    fn late_implicit(self, _: &u8, _: &u8) {}
    fn early<'a, 'b>(self) -> (&'a u8, &'b u8) { loop {} }
    fn late_early<'a, 'b>(self, _: &'a u8) -> &'b u8 { loop {} }
    fn late_implicit_early<'b>(self, _: &u8) -> &'b u8 { loop {} }
    fn late_implicit_self_early<'b>(&self) -> &'b u8 { loop {} }
    fn late_unused_early<'a, 'b>(self) -> &'b u8 { loop {} }
    fn life_and_type<'a, T>(self) -> &'a T { loop {} }
}

fn method_call() {
    S.early(); // OK
    S.early::<'static>();
    //~^ ERROR method takes 2 lifetime arguments but 1 lifetime argument
    S.early::<'static, 'static, 'static>();
    //~^ ERROR method takes 2 lifetime arguments but 3 lifetime arguments were supplied
    let _: &u8 = S.life_and_type::<'static>();
    S.life_and_type::<u8>();
    S.life_and_type::<'static, u8>();
}

fn ufcs() {
    S::late(S, &0, &0); // OK
    S::late::<'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late::<'static, 'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late::<'static, 'static, 'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_early(S, &0); // OK
    S::late_early::<'static, 'static>(S, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_early::<'static, 'static, 'static>(S, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly

    S::late_implicit(S, &0, &0); // OK
    S::late_implicit::<'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_implicit::<'static, 'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_implicit::<'static, 'static, 'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_implicit_early(S, &0); // OK
    S::late_implicit_early::<'static, 'static>(S, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_implicit_early::<'static, 'static, 'static>(S, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_implicit_self_early(&S); // OK
    S::late_implicit_self_early::<'static, 'static>(&S);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_implicit_self_early::<'static, 'static, 'static>(&S);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_unused_early(S); // OK
    S::late_unused_early::<'static, 'static>(S);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_unused_early::<'static, 'static, 'static>(S);
    //~^ ERROR cannot specify lifetime arguments explicitly

    S::early(S); // OK
    S::early::<'static>(S);
    //~^ ERROR method takes 2 lifetime arguments but 1 lifetime argument
    S::early::<'static, 'static, 'static>(S);
    //~^ ERROR method takes 2 lifetime arguments but 3 lifetime arguments were supplied
    let _: &u8 = S::life_and_type::<'static>(S);
    S::life_and_type::<u8>(S);
    S::life_and_type::<'static, u8>(S);
}

fn main() {}
