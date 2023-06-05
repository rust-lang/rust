macro_rules! borrow {
    ($x:expr) => { &$x }
}

fn foo(_: String) {}

fn foo2(s: &String) {
    foo(s);
    //~^ ERROR mismatched types
}

fn foo3(_: u32) {}
fn foo4(u: &u32) {
    foo3(u);
    //~^ ERROR mismatched types
}

struct S<'a> {
    u: &'a u32,
}

struct R {
    i: u32,
}

fn main() {
    let s = String::new();
    let r_s = &s;
    foo2(r_s);
    foo(&"aaa".to_owned());
    //~^ ERROR mismatched types
    foo(&mut "aaa".to_owned());
    //~^ ERROR mismatched types
    foo3(borrow!(0));
    //~^ ERROR mismatched types
    foo4(&0);
    assert_eq!(3i32, &3i32);
    //~^ ERROR mismatched types
    let u = 3;
    let s = S { u };
    //~^ ERROR mismatched types
    let s = S { u: u };
    //~^ ERROR mismatched types
    let i = &4;
    let r = R { i };
    //~^ ERROR mismatched types
    let r = R { i: i };
    //~^ ERROR mismatched types


    let a = &1;
    let b = &2;
    let val: i32 = if true {
        a + 1
    } else {
        b
        //~^ ERROR mismatched types
    };
    let val: i32 = if true {
        let _ = 2;
        a + 1
    } else {
        let _ = 2;
        b
        //~^ ERROR mismatched types
    };
    let val = if true {
        *a
    } else if true {
    //~^ ERROR incompatible types
        b
    } else {
        &0
    };

    #[derive(PartialEq, Eq)]
    struct Foo;
    let foo = Foo;
    let bar = &Foo;

    if foo == bar {
    //~^ ERROR mismatched types
    }
}
