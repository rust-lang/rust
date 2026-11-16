fn test(_x: &mut String) {}

fn test2(_x: &mut i32) {}

fn test3(_x: *mut String) {}

fn main() {
    let x: usize = String::new();
    //~^ ERROR E0308
    let x: &str = String::new();
    //~^ ERROR E0308
    let y = String::new();
    test(&y);
    //~^ ERROR E0308
    test(&raw const y);
    //~^ ERROR E0308
    test(&raw mut y);
    //~^ ERROR E0308
    test2(&y);
    //~^ ERROR E0308
    let s = &mut String::new();
    s = format!("foo");
    //~^ ERROR E0308
    let s = String::new();
    test3(&raw const s);
    //~^ ERROR E0308
    test3(&s);
    //~^ ERROR E0308
}
