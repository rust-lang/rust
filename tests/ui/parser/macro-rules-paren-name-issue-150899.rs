macro_rules!(i_think_the_name_should_go_here) {
    //~^ ERROR macros that expand to items must be delimited with braces or followed by a semicolon
    //~| ERROR expected item, found `{`
    () => {}
}

fn main() {}
