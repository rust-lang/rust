// Issue #97583.
trait Super<T> {
    type Assoc;
}
trait Trait: Super<u32> + Super<i32> {}

fn foo<T>(_: dyn Trait<Assoc = u32>, _: T) {} //~ ERROR: ambiguous associated type

fn main() {
    let _: dyn Trait<Assoc = u32>; //~ ERROR: ambiguous associated type
}

mod with_args_before {
    mod module { pub trait A<X> { type T; } }
    trait B<X>: module::A<(X,)> { type T; }
    trait C<X>: B<(X,)> {}

    fn f(_: impl C<bool, T = ()>) {} //~ ERROR: ambiguous associated type
    //fn f(_: impl C<bool> + B<(bool,), T = ()> + module::A<((bool,)), T = ()>) {}
}

mod with_args_after {
    mod module { pub trait A<X> { type T; } }
    trait B<X>: module::A<(X,)> { type T; }
    trait C<X>: B<(X,)> {}

    fn f(_: impl C<T = (), bool>) {} //~ ERROR: ambiguous associated type
    //~^ ERROR: generic arguments must come before the first constraint
    //fn f(_: impl C<bool> + B<(bool,), T = ()> + module::A<((bool,)), T = ()>) {}
}

mod with_args_mixed {
    mod module { pub trait A<X> { type T; } }
    trait B<X>: module::A<(X,)> { type T; }
    trait C<X, Y>: B<(X,)> {}

    fn f(_: impl C<bool, T = (), bool>) {} //~ ERROR: ambiguous associated type
    //~^ ERROR: generic arguments must come before the first constraint
}

mod with_args_mingled {
    mod module { pub trait A<X> { type T; type K;} }
    trait B<X>: module::A<(X,)> { type T; type K;}
    trait C<X, Y>: B<(X,)> {}

    fn f(_: impl C<bool, T = (), bool, K = ()>) {} //~ ERROR: ambiguous associated type
    //~^ ERROR: ambiguous associated type
    //~| ERROR: generic arguments must come before the first constraint
}

mod without_args_and_padded_first {
    trait A { type T; }
    trait B: A { type T; }
    trait C: B { type PadLeft; type PadRight; }

    fn f(_: impl C<T = (), PadLeft = (), PadRight = ()>) {} //~ ERROR: ambiguous associated type
    // fn f(_: impl C<PadLeft = (), PadRight = ()> + B<T = ()> + A<T = ()>) {}
}

mod without_args_and_padded_middle {
    trait A { type T; }
    trait B: A { type T; }
    trait C: B { type PadLeft; type PadRight; }

    fn f(_: impl C<PadLeft = (), T = (), PadRight = ()>) {} //~ ERROR: ambiguous associated type
    // fn f(_: impl C<PadLeft = (), PadRight = ()> + B<T = ()> + A<T = ()>) {}
}

mod without_args_and_padded_last {
    trait A { type T; }
    trait B: A { type T; }
    trait C: B { type PadLeft; type PadRight; }

    fn f(_: impl C<PadLeft = (), PadRight = (), T = ()>) {} //~ ERROR: ambiguous associated type
    // fn f(_: impl C<PadLeft = (), PadRight = ()> + B<T = ()> + A<T = ()>) {}
}

mod gat {
    // FIXME: if the arity of the two assoc tys differ, then what happens?

    trait A { type T<Q, K>; }
    trait B: A { type T<Q>; }
    trait C: B {}

    fn f(_: impl C<T<i32> = ()>) {} //~ ERROR: ambiguous associated type
    // The suggestion is incorrect.
    // fn f(_: impl C + B<T<i32> = ()> + A<T<i32> = ()>) {}
}

mod super_with_projs {
    trait A { type T; type Unrelated; }
    trait K { type T; }
    // Explainer: It's possible that we have ambiguous assoc item constraints despite all but one
    // already being constrained.
    trait B: A<Unrelated = String> + K<T = u32> {} // NOTE: not mentioning Unrelated in the sugg is perfect!!

    // fn t(_: impl B<> + A<T = ()>) {}
    fn f(_: impl B<T = ()>) {} //~ ERROR: ambiguous associated type
    // fn sugg(_: impl B<> + K<T = ()> + A<T = ()>) {} // FIXME: incorrect sugg
    //                     //^^^^^^^^^ we should skip this smh
}

// Here, the ident is ambiguous in the toplevel traitref but
// all assoc item candidates are constrained already meaning
// `disambiguated_bounds` will be empty once we filter out
// already-constrained bounds. therefore we need to emit a
// different suggestion: namely only remove!!!
// SOLUTION: repl `if d_bounds.empty.not { let constr=constr.unwrap`
//               ---> `iflet Some(constr) = constr { [[only sugg removal]]`
mod super_with_exhaustive_projs {
    trait A0 { type T; }
    trait A1 { type T; }
    trait B: A0<T = i32> + A1<T = u32> {} // NOTE: not mentioning Unrelated in the sugg is perfect!!

    // fn t(_: impl B) {}
    fn f(_: impl B<T = ()>) {} //~ ERROR: ambiguous associated type
    // fn sugg(_: impl B<> + A0<T = ()> + A<T = ()>) {} // FIXME: incorrect sugg
}

mod paren_sugar {
    trait A { type Output; }
    /*tmp*/trait FnOnce_ { type Output; }
    // BIG NOTE!!!! Unrelated to paren_sugar: It's possible that we have ambiguous assoc item
    // constraints despite all but one already being constrained.
    trait B: A + FnOnce() {}

    fn f(_: impl B<Output = ()>) {} //~ ERROR: ambiguous associated type
    // fn t(_: impl B + A<Output = ()>) {}
}

// Making sure we print for<> binders
mod higher_rank {
    trait A<'a> {type T;}
    trait B<'a> {type T;}
    trait C: for<'a> A<'a> + for<'r> B<'r> {}

    fn f(_: impl C<T = ()>) {} //~ ERROR: ambiguous associated type
    // fn f(_: impl C<> + for<'r> B<'r, T = ()> + for<'a> A<'a, T = ()>) {}
}

mod atb {
    trait A { type T; }
    trait B: A { type T; }
    trait C: B { }

    fn f(_: impl C<T: Copy>) {} //~ ERROR: ambiguous associated type
    // fn f(_: impl C + B<T: Copy> + A<T: Copy>) {}
}
