#![allow(unused, clippy::diverging_sub_expression, clippy::needless_if)]
#![warn(clippy::nonminimal_bool)]

fn methods_with_negation() {
    let a: Option<i32> = unimplemented!();
    let b: Result<i32, i32> = unimplemented!();
    let _ = a.is_some();
    let _ = !a.is_some();
    //~^ nonminimal_bool
    let _ = a.is_none();
    let _ = !a.is_none();
    //~^ nonminimal_bool
    let _ = b.is_err();
    let _ = !b.is_err();
    //~^ nonminimal_bool
    let _ = b.is_ok();
    let _ = !b.is_ok();
    //~^ nonminimal_bool
    let c = false;
    let _ = !(a.is_some() && !c);
    //~^ nonminimal_bool
    let _ = !(a.is_some() || !c);
    //~^ nonminimal_bool
    let _ = !(!c ^ c) || !a.is_some();
    //~^ nonminimal_bool
    let _ = (!c ^ c) || !a.is_some();
    //~^ nonminimal_bool
    let _ = !c ^ c || !a.is_some();
    //~^ nonminimal_bool
}

// Simplified versions of https://github.com/rust-lang/rust-clippy/issues/2638
// clippy::nonminimal_bool should only check the built-in Result and Some type, not
// any other types like the following.
enum CustomResultOk<E> {
    Ok,
    Err(E),
}
enum CustomResultErr<E> {
    Ok,
    Err(E),
}
enum CustomSomeSome<T> {
    Some(T),
    None,
}
enum CustomSomeNone<T> {
    Some(T),
    None,
}

impl<E> CustomResultOk<E> {
    pub fn is_ok(&self) -> bool {
        true
    }
}

impl<E> CustomResultErr<E> {
    pub fn is_err(&self) -> bool {
        true
    }
}

impl<T> CustomSomeSome<T> {
    pub fn is_some(&self) -> bool {
        true
    }
}

impl<T> CustomSomeNone<T> {
    pub fn is_none(&self) -> bool {
        true
    }
}

fn dont_warn_for_custom_methods_with_negation() {
    let res = CustomResultOk::Err("Error");
    // Should not warn and suggest 'is_err()' because the type does not
    // implement is_err().
    if !res.is_ok() {}

    let res = CustomResultErr::Err("Error");
    // Should not warn and suggest 'is_ok()' because the type does not
    // implement is_ok().
    if !res.is_err() {}

    let res = CustomSomeSome::Some("thing");
    // Should not warn and suggest 'is_none()' because the type does not
    // implement is_none().
    if !res.is_some() {}

    let res = CustomSomeNone::Some("thing");
    // Should not warn and suggest 'is_some()' because the type does not
    // implement is_some().
    if !res.is_none() {}
}

// Only Built-in Result and Some types should suggest the negated alternative
fn warn_for_built_in_methods_with_negation() {
    let res: Result<usize, usize> = Ok(1);
    if !res.is_ok() {}
    //~^ nonminimal_bool
    if !res.is_err() {}
    //~^ nonminimal_bool

    let res = Some(1);
    if !res.is_some() {}
    //~^ nonminimal_bool
    if !res.is_none() {}
    //~^ nonminimal_bool
}

#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn dont_warn_for_negated_partial_ord_comparison() {
    let a: f64 = unimplemented!();
    let b: f64 = unimplemented!();
    let _ = !(a < b);
    let _ = !(a <= b);
    let _ = !(a > b);
    let _ = !(a >= b);
}

fn issue_12625() {
    let a = 0;
    let b = 0;
    if !(a as u64 >= b) {}
    //~^ nonminimal_bool
    if !((a as u64) >= b) {}
    //~^ nonminimal_bool
    if !(a as u64 <= b) {}
    //~^ nonminimal_bool
}

fn issue_12761() {
    let a = 0;
    let b = 0;
    let c = 0;
    if !(a >= b) as i32 == c {}
    //~^ nonminimal_bool
    if !(a >= b) | !(a <= c) {}
    //~^ nonminimal_bool
    //~| nonminimal_bool
    let opt: Option<usize> = Some(1);
    let res: Result<usize, usize> = Ok(1);
    if !res.is_ok() as i32 == c {}
    //~^ nonminimal_bool
    if !res.is_ok() | !opt.is_none() {}
    //~^ nonminimal_bool
    //~| nonminimal_bool

    fn a(a: bool) -> bool {
        (!(4 > 3)).b()
        //~^ nonminimal_bool
    }

    trait B {
        fn b(&self) -> bool {
            true
        }
    }

    impl B for bool {}
}

fn issue_13436() {
    fn not_zero(x: i32) -> bool {
        x != 0
    }

    let opt = Some(500);
    _ = opt.is_some_and(|x| x < 1000);
    _ = opt.is_some_and(|x| x <= 1000);
    _ = opt.is_some_and(|x| x > 1000);
    _ = opt.is_some_and(|x| x >= 1000);
    _ = opt.is_some_and(|x| x == 1000);
    _ = opt.is_some_and(|x| x != 1000);
    _ = opt.is_some_and(not_zero);
    _ = !opt.is_some_and(|x| x < 1000);
    //~^ nonminimal_bool
    _ = !opt.is_some_and(|x| x <= 1000);
    //~^ nonminimal_bool
    _ = !opt.is_some_and(|x| x > 1000);
    //~^ nonminimal_bool
    _ = !opt.is_some_and(|x| x >= 1000);
    //~^ nonminimal_bool
    _ = !opt.is_some_and(|x| x == 1000);
    //~^ nonminimal_bool
    _ = !opt.is_some_and(|x| x != 1000);
    //~^ nonminimal_bool
    _ = !opt.is_some_and(not_zero);
    _ = opt.is_none_or(|x| x < 1000);
    _ = opt.is_none_or(|x| x <= 1000);
    _ = opt.is_none_or(|x| x > 1000);
    _ = opt.is_none_or(|x| x >= 1000);
    _ = opt.is_none_or(|x| x == 1000);
    _ = opt.is_none_or(|x| x != 1000);
    _ = opt.is_none_or(not_zero);
    _ = !opt.is_none_or(|x| x < 1000);
    //~^ nonminimal_bool
    _ = !opt.is_none_or(|x| x <= 1000);
    //~^ nonminimal_bool
    _ = !opt.is_none_or(|x| x > 1000);
    //~^ nonminimal_bool
    _ = !opt.is_none_or(|x| x >= 1000);
    //~^ nonminimal_bool
    _ = !opt.is_none_or(|x| x == 1000);
    //~^ nonminimal_bool
    _ = !opt.is_none_or(|x| x != 1000);
    //~^ nonminimal_bool
    _ = !opt.is_none_or(not_zero);

    let opt = Some(true);
    _ = opt.is_some_and(|x| x);
    _ = opt.is_some_and(|x| !x);
    _ = !opt.is_some_and(|x| x);
    _ = !opt.is_some_and(|x| !x);
    //~^ nonminimal_bool
    _ = opt.is_none_or(|x| x);
    _ = opt.is_none_or(|x| !x);
    _ = !opt.is_none_or(|x| x);
    _ = !opt.is_none_or(|x| !x);
    //~^ nonminimal_bool

    let opt: Option<Result<i32, i32>> = Some(Ok(123));
    _ = opt.is_some_and(|x| x.is_ok());
    _ = opt.is_some_and(|x| x.is_err());
    _ = opt.is_none_or(|x| x.is_ok());
    _ = opt.is_none_or(|x| x.is_err());
    _ = !opt.is_some_and(|x| x.is_ok());
    //~^ nonminimal_bool
    _ = !opt.is_some_and(|x| x.is_err());
    //~^ nonminimal_bool
    _ = !opt.is_none_or(|x| x.is_ok());
    //~^ nonminimal_bool
    _ = !opt.is_none_or(|x| x.is_err());
    //~^ nonminimal_bool

    #[clippy::msrv = "1.81"]
    fn before_stabilization() {
        let opt = Some(500);
        _ = !opt.is_some_and(|x| x < 1000);
    }
}

fn main() {}
