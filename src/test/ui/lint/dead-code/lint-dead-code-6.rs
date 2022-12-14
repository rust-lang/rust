#![deny(dead_code)]

struct UnusedStruct; //~ ERROR struct `UnusedStruct` is never constructed
impl UnusedStruct {
    fn unused_impl_fn_1() { //~ ERROR associated function `unused_impl_fn_1` is never used
        println!("blah");
    }

    fn unused_impl_fn_2(var: i32) { //~ ERROR associated function `unused_impl_fn_2` is never used
        println!("foo {}", var);
    }

    fn unused_impl_fn_3( //~ ERROR associated function `unused_impl_fn_3` is never used
        var: i32,
    ) {
        println!("bar {}", var);
    }
}

fn main() {}
