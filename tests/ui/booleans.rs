#![warn(clippy::nonminimal_bool, clippy::logic_bug)]

#[allow(unused, clippy::many_single_char_names)]
fn main() {
    let a: bool = unimplemented!();
    let b: bool = unimplemented!();
    let c: bool = unimplemented!();
    let d: bool = unimplemented!();
    let e: bool = unimplemented!();
    let _ = a && b || a;
    let _ = !(a && b);
    let _ = !true;
    let _ = !false;
    let _ = !!a;
    let _ = false && a;
    let _ = false || a;
    // don't lint on cfgs
    let _ = cfg!(you_shall_not_not_pass) && a;
    let _ = a || !b || !c || !d || !e;
    let _ = !(a && b || c);
    let _ = !(!a && b);
}

#[allow(unused, clippy::many_single_char_names)]
fn equality_stuff() {
    let a: i32 = unimplemented!();
    let b: i32 = unimplemented!();
    let c: i32 = unimplemented!();
    let d: i32 = unimplemented!();
    let e: i32 = unimplemented!();
    let _ = a == b && a != b;
    let _ = a == b && c == 5 && a == b;
    let _ = a == b && c == 5 && b == a;
    let _ = a < b && a >= b;
    let _ = a > b && a <= b;
    let _ = a > b && a == b;
    let _ = a != b || !(a != b || c == d);
}

#[allow(unused, clippy::many_single_char_names)]
fn methods_with_negation() {
    let a: Option<i32> = unimplemented!();
    let b: Result<i32, i32> = unimplemented!();
    let _ = a.is_some();
    let _ = !a.is_some();
    let _ = a.is_none();
    let _ = !a.is_none();
    let _ = b.is_err();
    let _ = !b.is_err();
    let _ = b.is_ok();
    let _ = !b.is_ok();
    let c = false;
    let _ = !(a.is_some() && !c);
    let _ = !(!c ^ c) || !a.is_some();
    let _ = (!c ^ c) || !a.is_some();
    let _ = !c ^ c || !a.is_some();
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
    if !res.is_err() {}

    let res = Some(1);
    if !res.is_some() {}
    if !res.is_none() {}
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

fn issue3847(a: u32, b: u32) -> bool {
    const THRESHOLD: u32 = 1_000;

    if a < THRESHOLD && b >= THRESHOLD || a >= THRESHOLD && b < THRESHOLD {
        return false;
    }
    true
}

fn issue4548() {
    fn f(_i: u32, _j: u32) -> u32 {
        unimplemented!();
    }

    let i = 0;
    let j = 0;

    if i != j && f(i, j) != 0 || i == j && f(i, j) != 1 {}
}
