// Check that closure debug implementation correctly displays all captures precisely.

//@ revisions: e2018 e2021
//@[e2018] edition: 2018
//@[e2021] edition: 2021

#![crate_type = "lib"]

pub fn foo(x: (String, String)) {
    // CHECK-LABEL: foo(
    // e2018: {closure{{.*}}issue_154166{{.*}}} { x: {{.*}} };
    // e2021: {closure{{.*}}issue_154166{{.*}}} { x__0: {{.*}}, x__1: {{.*}} };
    let _closure = || {
        if std::hint::black_box(true) {
            let _a = &x.1;
        } else {
            let _b = x.0;
        }
    };
}
