pub fn test_cloned(x: Option<i32>) {
    let _: i32 = x.cloned().unwrap();
    //~^ ERROR `Option<i32>` is not an iterator
}

pub fn test_copied(x: Option<i32>) {
    let _: i32 = x.copied().unwrap();
    //~^ ERROR `Option<i32>` is not an iterator
}

fn main() {}
