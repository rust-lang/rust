//@ compile-flags: -Zmir-opt-level=3
//@ run-pass

// Tests that specialization does not cause optimizations running on polymorphic MIR to resolve
// to a `default` implementation.

#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

trait Marker {}

trait SpecializedTrait {
    const CONST_BOOL: bool;
    const CONST_STR: &'static str;
    fn method() -> &'static str;
}
impl <T> SpecializedTrait for T {
    default const CONST_BOOL: bool = false;
    default const CONST_STR: &'static str = "in default impl";
    #[inline(always)]
    default fn method() -> &'static str {
        "in default impl"
    }
}
impl <T: Marker> SpecializedTrait for T {
    const CONST_BOOL: bool = true;
    const CONST_STR: &'static str = "in specialized impl";
    fn method() -> &'static str {
        "in specialized impl"
    }
}

fn const_bool<T>() -> &'static str {
    if <T as SpecializedTrait>::CONST_BOOL {
        "in specialized impl"
    } else {
        "in default impl"
    }
}
fn const_str<T>() -> &'static str {
    <T as SpecializedTrait>::CONST_STR
}
fn run_method<T>() -> &'static str {
    <T as SpecializedTrait>::method()
}

struct TypeA;
impl Marker for TypeA {}
struct TypeB;

#[inline(never)]
fn exit_if_not_eq(left: &str, right: &str) {
    if left != right {
        std::process::exit(1);
    }
}

pub fn main() {
    exit_if_not_eq("in specialized impl", const_bool::<TypeA>());
    exit_if_not_eq("in default impl", const_bool::<TypeB>());
    exit_if_not_eq("in specialized impl", const_str::<TypeA>());
    exit_if_not_eq("in default impl", const_str::<TypeB>());
    exit_if_not_eq("in specialized impl", run_method::<TypeA>());
    exit_if_not_eq("in default impl", run_method::<TypeB>());
}
