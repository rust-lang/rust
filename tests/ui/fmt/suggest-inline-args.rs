mod foo {
    pub fn bar() -> i32 {
        1
    }
}

fn bar() -> i32 {
    2
}

fn main() {
    let stderr = 3;
    eprintln!({stderr});
    //~^ ERROR format argument must be a string literal
    //~| HELP quote your inlined format argument to use as string literal
    eprintln!({1});
    //~^ ERROR format argument must be a string literal
    //~| HELP you might be missing a string literal to format with
    eprintln!({foo::bar()});
    //~^ ERROR format argument must be a string literal
    //~| HELP you might be missing a string literal to format with
    eprintln!({bar()});
    //~^ ERROR format argument must be a string literal
    //~| HELP you might be missing a string literal to format with
    eprintln!({1; 2});
    //~^ ERROR format argument must be a string literal
    //~| HELP you might be missing a string literal to format with
}
