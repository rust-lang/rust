struct S;

enum Age {
    Years(i64, i64)
}

fn foo() {
    let mut age = 29;
    Age::Years({age += 1; age}, 55)
    //~^ ERROR mismatched types
}

fn bar() {
    let mut age = 29;
    Age::Years(age, 55)
    //~^ ERROR mismatched types
}

fn baz() {
    S
    //~^ ERROR mismatched types
}

fn main() {}
