// Test that we cannot parse a closure with an explicit return type
// unless it uses braces.

fn needs_braces_1() {
    let x = || -> i32 22;
    //~^ ERROR expected `{`, found `22`
}

// Check other delimiters too.

fn needs_braces_2() {
        let x = || -> (i32, i32) (1, 2);
        //~^ ERROR expected `{`, found `(`
}

fn needs_braces_3() {
        let c = || -> [i32; 2] [1, 2];
        //~^ ERROR expected `{`, found `[`
}

fn main() {}
