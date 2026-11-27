fn foo() {
    let _ = 0..<10;
    //~^ ERROR: expected type, found `10`
    //~| HELP: remove the `<` to write an exclusive range
}

fn bar() {
    let _ = 0..<foo;
    //~^ ERROR: expected one of `!`, `(`, `+`, `::`, `<`, `>`, or `as`, found `;`
    //~| HELP: remove the `<` to write an exclusive range
}

fn baz() {
    let _ = <foo>;
    //~^ ERROR: expected `::`, found `;`
}

fn qux() {
    let _ = [1, 2, 3][..<1];
    //~^ ERROR: expected type, found `1`
    //~| HELP: remove the `<` to write an exclusive range
}

fn quux() {
    let _ = [1, 2, 3][..<foo];
    //~^ ERROR: expected one of `!`, `(`, `+`, `::`, `<`, `>`, or `as`, found `]`
    //~| HELP: remove the `<` to write an exclusive range
}

fn foobar() {
    let _ = [1, 2, 3][..<foo>];
    //~^ ERROR: expected `::`, found `]`
}

fn ok1() {
    let _ = [1, 2, 3][..<usize>::default()];
}

fn ok2() {
    let _ = 0..<i32>::default();
}

fn main() {}
