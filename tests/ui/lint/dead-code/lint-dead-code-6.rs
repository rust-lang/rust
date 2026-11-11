#![deny(dead_code)]

struct UnusedStruct; //~ ERROR struct `UnusedStruct` is never constructed
impl UnusedStruct {
    fn unused_impl_fn_1() {
        //~^ ERROR associated functions `unused_impl_fn_1`, `unused_impl_fn_2`, and `unused_impl_fn_3` are never used [dead_code]
        println!("blah");
    }

    fn unused_impl_fn_2(var: i32) {
        println!("foo {}", var);
    }

    fn unused_impl_fn_3(var: i32) {
        println!("bar {}", var);
    }
}

fn main() {}
