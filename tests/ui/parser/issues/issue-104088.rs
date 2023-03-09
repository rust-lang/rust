fn test() {
    if let 123 = 123 { println!("yes"); }
}

fn test_2() {
    let 1x = 123;
    //~^ ERROR expected identifier, found `1x`
}

fn test_3() {
    let 2x: i32 = 123;
    //~^ ERROR expected identifier, found `2x`
}

fn test_4() {
    if let 2e1 = 123 {
        //~^ ERROR mismatched types
    }
}

fn test_5() {
    let 23name = 123;
    //~^ ERROR expected identifier, found `23name`
}

fn main() {}
