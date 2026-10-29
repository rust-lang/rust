fn 1234test() {
//~^ ERROR expected identifier, found `1234test`
    if let 123 = 123 { println!("yes"); }

    if let 2e1 = 123 {
        //~^ ERROR mismatched types
    }

    let 23name = 123;
    //~^ ERROR expected identifier, found `23name`
}
fn foo() {
    let 2x: i32 = 123;
    //~^ ERROR expected identifier, found `2x`
}
fn bar() {
    let 1x = 123;
    //~^ ERROR expected identifier, found `1x`
}

fn main() {}
