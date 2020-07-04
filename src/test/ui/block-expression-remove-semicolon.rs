// run-rustfix

fn foo() -> i32 {
    0
}

fn main() {
    let _x: i32 = {
        //~^ ERROR mismatched types
        foo(); //~ HELP consider removing this semicolon
    };
}
