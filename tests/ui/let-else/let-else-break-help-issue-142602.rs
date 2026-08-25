// testcase from https://github.com/rust-lang/rust/issues/142602

pub fn main() {
    // Case 1: break before let-else
    let _a = loop {
        if true {
            break;
        }
        let Some(_) = Some(5) else {
            break 3; //~ ERROR mismatched types
        };
    };

    // Case 2: two let-else statements
    let _b = loop {
        let Some(_) = Some(5) else {
            break;
        };
        let Some(_) = Some(4) else {
            break 3; //~ ERROR mismatched types
        };
    };
}
