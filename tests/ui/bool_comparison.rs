#[warn(clippy::bool_comparison)]
fn main() {
    let x = true;
    if x == true {
        "yes"
    } else {
        "no"
    };
    if x == false {
        "yes"
    } else {
        "no"
    };
    if true == x {
        "yes"
    } else {
        "no"
    };
    if false == x {
        "yes"
    } else {
        "no"
    };
    if x != true {
        "yes"
    } else {
        "no"
    };
    if x != false {
        "yes"
    } else {
        "no"
    };
    if true != x {
        "yes"
    } else {
        "no"
    };
    if false != x {
        "yes"
    } else {
        "no"
    };
    if x < true {
        "yes"
    } else {
        "no"
    };
    if false < x {
        "yes"
    } else {
        "no"
    };
    if x > false {
        "yes"
    } else {
        "no"
    };
    if true > x {
        "yes"
    } else {
        "no"
    };
    let y = true;
    if x < y {
        "yes"
    } else {
        "no"
    };
    if x > y {
        "yes"
    } else {
        "no"
    };
}
