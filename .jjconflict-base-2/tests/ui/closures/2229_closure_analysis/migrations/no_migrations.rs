//@ run-pass

// Set of test cases that don't need migrations

#![deny(rust_2021_incompatible_closure_captures)]

// Copy types as copied by the closure instead of being moved into the closure
// Therefore their drop order isn't tied to the closure and won't be requiring any
// migrations.
fn test1_only_copy_types() {
    let t = (0i32, 0i32);

    let c = || {
        let _t = t.0;
    };

    c();
}

// Same as test1 but using a move closure
fn test2_only_copy_types_move_closure() {
    let t = (0i32, 0i32);

    let c = move || {
        println!("{}", t.0);
    };

    c();
}

// Don't need to migrate if captured by ref
fn test3_only_copy_types_move_closure() {
    let t = (String::new(), String::new());

    let c = || {
        println!("{}", t.0);
    };

    c();
}

// Test migration analysis in case of Insignificant Drop + Non Drop aggregates.
// Note in this test the closure captures a non Drop type and therefore the variable
// is only captured by ref.
fn test4_insignificant_drop_non_drop_aggregate() {
    let t = (String::new(), 0i32);

    let c = || {
        let _t = t.1;
    };

    c();
}

struct Foo(i32);
impl Drop for Foo {
    fn drop(&mut self) {
        println!("{:?} dropped", self.0);
    }
}

// Test migration analysis in case of Significant Drop + Non Drop aggregates.
// Note in this test the closure captures a non Drop type and therefore the variable
// is only captured by ref.
fn test5_significant_drop_non_drop_aggregate() {
    let t = (Foo(0), 0i32);

    let c = || {
        let _t = t.1;
    };

    c();
}

fn main() {
    test1_only_copy_types();
    test2_only_copy_types_move_closure();
    test3_only_copy_types_move_closure();
    test4_insignificant_drop_non_drop_aggregate();
    test5_significant_drop_non_drop_aggregate();
}
