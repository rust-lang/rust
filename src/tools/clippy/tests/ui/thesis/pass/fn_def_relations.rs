//@rustc-env: CLIPPY_PETS_TEST_RELATIONS=1
//@rustc-env: CLIPPY_PRINT_MIR=1

#[warn(clippy::borrow_pats)]
fn no_rep(_: u32, _: String) -> u16 {
    12
}

#[warn(clippy::borrow_pats)]
fn direct_dep(a: &String, _: u32) -> &String {
    a
}

#[warn(clippy::borrow_pats)]
fn lifetime_dep<'a>(_: &String, a: &'a String) -> &'a String {
    a
}

#[warn(clippy::borrow_pats)]
fn lifetime_dep_more<'a>(_: &'a String, a: &'a String) -> &'a String {
    a
}

#[warn(clippy::borrow_pats)]
fn lifetime_dep_or<'a>(a: &'a String, b: &'a String) -> &'a String {
    if true { a } else { b }
}

#[warn(clippy::borrow_pats)]
fn lifetime_dep_const<'a>(_: &'a str, a: &'a str) -> &'a str {
    a
}

#[warn(clippy::borrow_pats)]
fn lifetime_dep_prim<'a>(_: &'a u32, a: &'a u32) -> &'a u32 {
    a
}

#[warn(clippy::borrow_pats)]
fn lifetime_outlives_middle_1<'a, 'd: 'a, 'b: 'd, 'c: 'd>(b: &'b String, _c: &'c String) -> &'a String {
    b
}

#[warn(clippy::borrow_pats)]
fn lifetime_outlives_middle_2<'a, 'b: 'd, 'c: 'd, 'd: 'a>(a: &'b String, b: &'c String) -> &'a String {
    if true { a } else { b }
}

#[warn(clippy::borrow_pats)]
fn lifetime_outlives<'a, 'b: 'a, 'c: 'a>(b: &'b String, _c: &'c String) -> &'a String {
    b
}

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

#[warn(clippy::borrow_pats)]
fn immutable_owner(_owner: &Owner<'_>, _val: &String) {}

#[warn(clippy::borrow_pats)]
fn mutable_owner(_owner: &mut Owner<'_>, _val: &String) {}

#[warn(clippy::borrow_pats)]
fn mutable_owner_and_arg(_owner: &mut Owner<'_>, _val: &mut String) {}

#[warn(clippy::borrow_pats)]
fn mutable_owner_and_lifetimed_str<'a>(owner: &mut Owner<'a>, val: &'a mut String) {
    owner.val = val;
}

#[warn(clippy::borrow_pats)]
fn immutable_owner_and_lifetimed_str<'a>(_owner: &Owner<'a>, val: &'a mut String) {}

#[warn(clippy::borrow_pats)]
fn mutable_owner_other_lifetimed_str<'a, 'b>(_owner: &'b &mut Owner<'a>, _val: &'b mut String) {}

#[warn(clippy::borrow_pats)]
fn mutable_owner_self_lifetimed_str<'a>(owner: &'a mut Owner<'a>, val: &'a mut String) {
    owner.val = val;
}

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

#[warn(clippy::borrow_pats)]
fn test_return_regions_static() -> &'static str {
    "Ducks"
}

#[warn(clippy::borrow_pats)]
fn test_return_regions_non_static(_arg: &str) -> &str {
    "Ducks"
}

#[warn(clippy::borrow_pats)]
fn test_return_regions_non_static_or_default(arg: &str) -> &str {
    if arg.is_empty() { "Ducks" } else { arg }
}

#[warn(clippy::borrow_pats)]
fn test_return_static_tuple_for_non_static(arg: &str) -> (&str, &str) {
    if arg.is_empty() { ("hey", "you") } else { (arg, arg) }
}

#[warn(clippy::borrow_pats)]
fn test_return_static_tuple(arg: &str) -> (&'static str, &'static str) {
    if arg.is_empty() {
        ("hey", "you")
    } else {
        ("duck", "cat")
    }
}

fn main() {}
