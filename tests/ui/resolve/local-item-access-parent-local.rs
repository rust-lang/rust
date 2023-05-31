#![allow(unused_variables, dead_code)]

fn f2() {
    let foo = 0;
    fn f() {
        foo;
        //~^ ERROR can't capture dynamic
    }
    const C: () = {
        foo;
        //~^ ERROR attempt to
    };
    static S: () = {
        foo;
        //~^ ERROR attempt to
    };
}

fn main() {}
