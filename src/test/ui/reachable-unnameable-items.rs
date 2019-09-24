// run-pass
// ignore-wasm32-bare compiled with panic=abort by default
// aux-build:reachable-unnameable-items.rs

extern crate reachable_unnameable_items;
use reachable_unnameable_items::*;

fn main() {
    let res1 = function_returning_unnameable_type().method_of_unnameable_type1();
    let res2 = CONSTANT_OF_UNNAMEABLE_TYPE.method_of_unnameable_type2();
    let res4 = AliasOfUnnameableType{}.method_of_unnameable_type4();
    let res5 = function_returning_unnameable_type().inherent_method_returning_unnameable_type().
                                                    method_of_unnameable_type5();
    let res6 = function_returning_unnameable_type().trait_method_returning_unnameable_type().
                                                    method_of_unnameable_type6();
    let res7 = STATIC.field_of_unnameable_type.method_of_unnameable_type7();
    let res8 = generic_function::<AliasOfUnnameableType>().method_of_unnameable_type8();
    let res_enum = NameableVariant.method_of_unnameable_enum();
    assert_eq!(res1, "Hello1");
    assert_eq!(res2, "Hello2");
    assert_eq!(res4, "Hello4");
    assert_eq!(res5, "Hello5");
    assert_eq!(res6, "Hello6");
    assert_eq!(res7, "Hello7");
    assert_eq!(res8, "Hello8");
    assert_eq!(res_enum, "HelloEnum");

    let none = None;
    function_accepting_unnameable_type(none);
    let _guard = std::panic::catch_unwind(|| none.unwrap().method_of_unnameable_type3());
}
