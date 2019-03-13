fn foo() -> i32 {
   0
}

fn main() {
    let x: i32 = {
        //~^ ERROR mismatched types
        foo(); //~ HELP consider removing this semicolon
    };
}
