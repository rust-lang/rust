// Regression test for https://github.com/rust-lang/rust/issues/150040
// When a `let PAT;` has no explicit type, later assignments can infer a non-well-formed
// pattern type such as `[str; 2]` or `(str, i32)`. We must reject those array and tuple
// patterns instead of accepting the invalid type or causing ICE.

#![allow(unused)]

struct S<T: ?Sized>(T);

fn should_fail_1() {
    let ref y @ [ref x, _]; //~ ERROR E0277
    x = "";
}

fn should_fail_2() {
    let [ref x]; //~ ERROR E0277
    x = "";
}

fn should_fail_3() {
    let [[ref x], [_, y @ ..]]; //~ ERROR E0277
    x = "";
    y = [];
}

fn should_fail_4() {
    let [(ref a, b), x]; //~ ERROR E0277
    a = "";
    b = 5;
}

fn should_fail_5() {
    let (ref a, b); //~ ERROR E0277
    a = "";
    b = 5;
}

fn should_fail_6() {
    let [S(ref x)]; //~ ERROR E0277
    x = "";
}

fn should_pass_1() {
    let ref x;
    x = "";
}

fn should_pass_2() {
    let ref y @ (ref x,);
    x = "";
}

fn should_pass_3() {
    let S(ref x);
    x = "";
}

fn main() {}
