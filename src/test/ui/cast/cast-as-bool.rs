fn main() {
    let u = 5 as bool; //~ ERROR cannot cast as `bool`
                       //~| HELP compare with zero instead
                       //~| SUGGESTION 5 != 0
    let t = (1 + 2) as bool; //~ ERROR cannot cast as `bool`
                             //~| HELP compare with zero instead
                             //~| SUGGESTION (1 + 2) != 0
    let v = "hello" as bool; //~ ERROR cannot cast as `bool`
}
