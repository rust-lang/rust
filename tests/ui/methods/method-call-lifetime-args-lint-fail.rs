#![deny(late_bound_lifetime_arguments)]
#![allow(unused)]

struct S;

impl S {
    fn late<'a, 'b>(self, _: &'a u8, _: &'b u8) {}
    fn late_implicit(self, _: &u8, _: &u8) {}
    fn late_early<'a, 'b>(self, _: &'a u8) -> &'b u8 { loop {} }
    fn late_implicit_early<'b>(self, _: &u8) -> &'b u8 { loop {} }

    // 'late lifetimes here belong to nested types not to the tested functions.
    fn early_tricky_explicit<'a>(_: for<'late> fn(&'late u8),
                                 _: Box<dyn for<'late> Fn(&'late u8)>)
                                 -> &'a u8 { loop {} }
    fn early_tricky_implicit<'a>(_: fn(&u8),
                                 _: Box<dyn Fn(&u8)>)
                                 -> &'a u8 { loop {} }
}

fn method_call() {
    S.late(&0, &0); // OK
    S.late::<'static>(&0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
    S.late::<'static, 'static>(&0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
    S.late::<'static, 'static, 'static>(&0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
    S.late_early(&0); // OK
    S.late_early::<'static>(&0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
    S.late_early::<'static, 'static>(&0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
    S.late_early::<'static, 'static, 'static>(&0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted

    S.late_implicit(&0, &0); // OK
    S.late_implicit::<'static>(&0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
    S.late_implicit::<'static, 'static>(&0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
    S.late_implicit::<'static, 'static, 'static>(&0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
    S.late_implicit_early(&0); // OK
    S.late_implicit_early::<'static>(&0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
    S.late_implicit_early::<'static, 'static>(&0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
    S.late_implicit_early::<'static, 'static, 'static>(&0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted

    S::early_tricky_explicit::<'static>(loop {}, loop {}); // OK
    S::early_tricky_implicit::<'static>(loop {}, loop {}); // OK
}

fn ufcs() {
    S::late_early::<'static>(S, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted

    S::late_implicit_early::<'static>(S, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
}

fn lint_not_inference_error() {
    fn f<'early, 'late, T: 'early>() {}

    // Make sure `u8` is substituted and not replaced with an inference variable
    f::<'static, u8>;
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
}

fn main() {}
