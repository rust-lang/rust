// run-rustfix
#![deny(disjoint_capture_migration)]
//~^ NOTE: the lint level is defined here

// Test cases for types that implement a significant drop (user defined)

#[derive(Debug)]
struct Foo(i32);
impl Drop for Foo {
    fn drop(&mut self) {
        println!("{:?} dropped", self.0);
    }
}

#[derive(Debug)]
struct ConstainsDropField(Foo, Foo);

// `t` needs Drop because one of its elements needs drop,
// therefore precise capture might affect drop ordering
fn test1_all_need_migration() {
    let t = (Foo(0), Foo(0));
    let t1 = (Foo(0), Foo(0));
    let t2 = (Foo(0), Foo(0));

    let c = || {
    //~^ ERROR: drop order affected for closure because of `capture_disjoint_fields`
    //~| HELP: add a dummy let to cause `t`, `t1`, `t2` to be fully captured
        let _t = t.0;
        let _t1 = t1.0;
        let _t2 = t2.0;
    };

    c();
}

// String implements drop and therefore should be migrated.
// But in this test cases, `t2` is completely captured and when it is dropped won't be affected
fn test2_only_precise_paths_need_migration() {
    let t = (Foo(0), Foo(0));
    let t1 = (Foo(0), Foo(0));
    let t2 = (Foo(0), Foo(0));

    let c = || {
    //~^ ERROR: drop order affected for closure because of `capture_disjoint_fields`
    //~| HELP: add a dummy let to cause `t`, `t1` to be fully captured
        let _t = t.0;
        let _t1 = t1.0;
        let _t2 = t2;
    };

    c();
}

// If a variable would've not been captured by value then it would've not been
// dropped with the closure and therefore doesn't need migration.
fn test3_only_by_value_need_migration() {
    let t = (Foo(0), Foo(0));
    let t1 = (Foo(0), Foo(0));
    let c = || {
    //~^ ERROR: drop order affected for closure because of `capture_disjoint_fields`
    //~| HELP: add a dummy let to cause `t` to be fully captured
        let _t = t.0;
        println!("{:?}", t1.1);
    };

    c();
}

// The root variable might not implement drop themselves but some path starting
// at the root variable might implement Drop.
//
// If this path isn't captured we need to migrate for the root variable.
fn test4_type_contains_drop_need_migration() {
    let t = ConstainsDropField(Foo(0), Foo(0));

    let c = || {
    //~^ ERROR: drop order affected for closure because of `capture_disjoint_fields`
    //~| HELP: add a dummy let to cause `t` to be fully captured
        let _t = t.0;
    };

    c();
}

// Test migration analysis in case of Drop + Non Drop aggregates.
// Note we need migration here only because the non-copy (because Drop type) is captured,
// otherwise we won't need to, since we can get away with just by ref capture in that case.
fn test5_drop_non_drop_aggregate_need_migration() {
    let t = (Foo(0), Foo(0), 0i32);

    let c = || {
    //~^ ERROR: drop order affected for closure because of `capture_disjoint_fields`
    //~| HELP: add a dummy let to cause `t` to be fully captured
        let _t = t.0;
    };

    c();
}

// Test migration analysis in case of Significant and Insignificant Drop aggregates.
fn test6_significant_insignificant_drop_aggregate_need_migration() {
    let t = (Foo(0), String::new());

    let c = || {
    //~^ ERROR: drop order affected for closure because of `capture_disjoint_fields`
    //~| HELP: add a dummy let to cause `t` to be fully captured
        let _t = t.1;
    };

    c();
}

// Since we are using a move closure here, both `t` and `t1` get moved
// even though they are being used by ref inside the closure.
fn test7_move_closures_non_copy_types_might_need_migration() {
    let t = (Foo(0), Foo(0));
    let t1 = (Foo(0), Foo(0), Foo(0));

    let c = move || {
    //~^ ERROR: drop order affected for closure because of `capture_disjoint_fields`
    //~| HELP: add a dummy let to cause `t1`, `t` to be fully captured
        println!("{:?} {:?}", t1.1, t.1);
    };

    c();
}

fn main() {
    test1_all_need_migration();
    test2_only_precise_paths_need_migration();
    test3_only_by_value_need_migration();
    test4_type_contains_drop_need_migration();
    test5_drop_non_drop_aggregate_need_migration();
    test6_significant_insignificant_drop_aggregate_need_migration();
    test7_move_closures_non_copy_types_might_need_migration();
}
