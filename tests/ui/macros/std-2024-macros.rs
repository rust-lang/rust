// Tests a small handful of macros in the standard library how they handle the
// new behavior introduced in 2024 that allows `const{}` expressions.

fn main() {
    assert_eq!(0, const { 0 });
    //~^ ERROR: no rules expected keyword `const`
    assert_eq!(const { 0 }, const { 0 });
    //~^ ERROR: no rules expected keyword `const`
    assert_eq!(const { 0 }, 0);
    //~^ ERROR: no rules expected keyword `const`

    let _: Vec<Vec<String>> = vec![const { vec![] }];
    //~^ ERROR: no rules expected keyword `const`
    let _: Vec<Vec<String>> = vec![const { vec![] }; 10];
    //~^ ERROR: no rules expected keyword `const`
}
