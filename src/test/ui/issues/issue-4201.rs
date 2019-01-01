fn main() {
    let a = if true {
        0
    } else if false {
//~^ ERROR if may be missing an else clause
//~| expected type `()`
//~| found type `{integer}`
//~| expected (), found integer
        1
    };
}
