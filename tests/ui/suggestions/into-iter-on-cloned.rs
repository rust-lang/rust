pub fn test_cloned(x: Option<i32>) {
    let _: i32 = x.cloned().unwrap();
    //~^ ERROR no method named `cloned`
    //~| HELP delete the call to `cloned`
}

pub fn test_copied(x: Option<i32>) {
    let _: i32 = x.copied().unwrap();
    //~^ ERROR no method named `copied`
    //~| HELP delete the call to `copied`
}

fn main() {}
