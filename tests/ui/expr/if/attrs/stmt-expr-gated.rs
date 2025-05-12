fn main() {
    let _ = #[deny(warnings)] if true { //~ ERROR attributes on expressions
    } else if false {
    } else {
    };
}
