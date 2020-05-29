#![deny(dead_code)]

struct UnusedStruct; //~ ERROR struct is never constructed: `UnusedStruct`
impl UnusedStruct {
    fn unused_impl_fn_1() { //~ ERROR associated function is never used: `unused_impl_fn_1`
        println!("blah");
    }

    fn unused_impl_fn_2(var: i32) { //~ ERROR associated function is never used: `unused_impl_fn_2`
        println!("foo {}", var);
    }

    fn unused_impl_fn_3( //~ ERROR associated function is never used: `unused_impl_fn_3`
        var: i32,
    ) {
        println!("bar {}", var);
    }
}

fn main() {}
