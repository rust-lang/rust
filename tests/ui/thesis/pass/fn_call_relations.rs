//@rustc-env: CLIPPY_PETS_TEST_RELATIONS=1
//@rustc-env: CLIPPY_PRINT_MIR=1

//! A file to test dependencies between function parameters
#![allow(unused)]

fn no_rep(_: u32, _: String) -> u16 {
    12
}

fn direct_dep(a: &String, _: u32) -> &String {
    a
}

fn lifetime_dep<'a>(_: &String, a: &'a String) -> &'a String {
    a
}

fn lifetime_dep_more<'a>(_: &'a String, a: &'a String) -> &'a String {
    a
}

fn lifetime_dep_const<'a>(_: &'a str, a: &'a str) -> &'a str {
    a
}

fn lifetime_dep_prim<'a>(_: &'a u32, a: &'a u32) -> &'a u32 {
    a
}

fn lifetime_outlives_middle_1<'a, 'd: 'a, 'b: 'd, 'c: 'd>(b: &'b String, _c: &'c String) -> &'a String {
    b
}

fn lifetime_outlives_middle_2<'a, 'b: 'd, 'c: 'd, 'd: 'a>(b: &'b String, _c: &'c String) -> &'a String {
    b
}

fn lifetime_outlives<'a, 'b: 'a, 'c: 'a>(b: &'b String, _c: &'c String) -> &'a String {
    b
}

#[warn(clippy::borrow_pats)]
fn test_return(s1: String, s2: String, s3: String, pair: (String, String)) {
    let _test = no_rep(1, s3);
    let _test = direct_dep(&s1, 2);
    let _test = lifetime_dep(&s1, &s2);
    let _test = lifetime_dep_more(&s1, &s2);
    let _test = lifetime_dep_prim(&1, &2);
    let _test = lifetime_dep_const("In", "Put");
    let _test = lifetime_outlives_middle_1(&s1, &s2);
    let _test = lifetime_outlives_middle_2(&s1, &s2);
    let _test = lifetime_outlives(&s1, &s2);
    let _test = lifetime_outlives(&pair.0, &pair.1);
}

struct Owner<'a> {
    val: &'a String,
}

fn immutable_owner(_owner: &Owner<'_>, _val: &String) {}

fn mutable_owner(_owner: &mut Owner<'_>, _val: &String) {}

fn mutable_owner_and_arg(_owner: &mut Owner<'_>, _val: &mut String) {}

fn mutable_owner_and_lifetimed_str<'a>(owner: &mut Owner<'a>, val: &'a mut String) {
    owner.val = val;
}

fn immutable_owner_and_lifetimed_str<'a>(_owner: &Owner<'a>, val: &'a mut String) {}

fn mutable_owner_other_lifetimed_str<'a, 'b>(_owner: &'b &mut Owner<'a>, _val: &'b mut String) {}

fn mutable_owner_self_lifetimed_str<'a>(_owner: &'a mut Owner<'a>, _val: &'a mut String) {}

#[warn(clippy::borrow_pats)]
fn test_mut_args_1<'a>(owner: &mut Owner<'a>, value: &'a mut String) {
    immutable_owner(owner, value);
    mutable_owner(owner, value);
    mutable_owner_and_arg(owner, value);
    mutable_owner_and_lifetimed_str(owner, value);
}

#[warn(clippy::borrow_pats)]
fn test_mut_args_2<'a>(owner: &mut Owner<'a>, value: &'a mut String) {
    mutable_owner_other_lifetimed_str(&owner, value);
}

#[warn(clippy::borrow_pats)]
fn test_mut_args_3<'a>(owner: &'a mut Owner<'a>, value: &'a mut String) {
    mutable_owner_self_lifetimed_str(owner, value);
}

fn take_string(_s: String) {}
fn take_string_ref(_s: &String) {}
fn pass_t<T>(tee: T) -> T {
    tee
}

#[warn(clippy::borrow_pats)]
fn use_local_func() {
    let func = take_string_ref;

    func(&String::new());
}

#[warn(clippy::borrow_pats)]
fn use_arg_func(func: fn(&String) -> &str) {
    func(&String::new());
}

#[warn(clippy::borrow_pats)]
fn borrow_as_generic(s: String) {
    let _tee = pass_t(&s);
    // let _len = tee.len();
}

fn main() {}
