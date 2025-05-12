//@ edition:2021

enum SingleVariant {
    A
}

struct TestStruct {
    x: i32,
    y: i32,
    z: i32,
}

fn edge_case_if() {
    let sv = SingleVariant::A;
    let condition = true;
    // sv should not be captured as it is a SingleVariant
    let _a = || {
        match sv {
            SingleVariant::A if condition => (),
            _ => ()
        }
    };
    let mut mut_sv = sv;
    _a();

    // ts should be captured
    let ts = TestStruct { x: 1, y: 1, z: 1 };
    let _b = || { match ts {
        TestStruct{ x: 1, .. } => (),
        _ => ()
    }};
    let mut mut_ts = ts;
    //~^ ERROR: cannot move out of `ts` because it is borrowed
    _b();
}

fn main() {}
