//@rustc-env: CLIPPY_PRINT_MIR=1

#![allow(unused)]

fn magic_1<T>(b: &T) {}
fn magic_2<T>(b: &T, c: &T) {}

fn print_mir() {
    let a = if true {
        let mut x = Vec::new();
        x.push(1);
        x
    } else {
        let mut y = Vec::new();
        y.push(89);
        y
    };
}

struct A {
    field: String,
}

impl A {
    fn borrow_field_direct(&self) -> &String {
        &self.field
    }

    fn borrow_field_deref(&self) -> &str {
        &self.field
    }
}

// fn simple_ownership(cond: bool) {
//     let a = String::new();
//     let b = String::new();
//
//     let x;
//     if cond {
//         x = &a;
//     } else {
//         x = &b;
//     }
//
//     magic_1(x);
// }

// fn if_fun(a: String, b: String, cond: bool) {
//     if cond {
//         magic_1(&a);
//     } else {
//         magic_1(&b);
//     }
//
//     magic_1(if cond {&a} else {&b});
// }

fn main() {}
