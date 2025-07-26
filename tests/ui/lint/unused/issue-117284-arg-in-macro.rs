#![deny(unused_variables)]
macro_rules! make_var {
    ($struct:ident, $var:ident) => {
        let $var = $struct.$var; //~ ERROR unused variable: `var`
    };
}

#[allow(unused)]
struct MyStruct {
    var: i32,
}

fn main() {
    let s = MyStruct { var: 42 };
    make_var!(s, var);
    let a = 1; //~ ERROR unused variable: `a`
}
