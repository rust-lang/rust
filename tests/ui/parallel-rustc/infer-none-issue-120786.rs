// Test for #120786, which causes a deadlock bug
//
//@ parallel-front-end-robustness
//@ compile-flags: -Z threads=50

fn no_err() {
    |x: u32, y| x;
    let _ = String::from("x");
}

fn err() {
    String::from("x".as_ref());
}

fn arg_pat_closure_err() {
    |x| String::from("x".as_ref());
}

fn local_pat_closure_err() {
    let _ = "x".as_ref();
}

fn err_first_arg_pat() {
    String::from("x".as_ref());
    |x: String| x;
}

fn err_second_arg_pat() {
    |x: String| x;
    String::from("x".as_ref());
}

fn err_mid_arg_pat() {
    |x: String| x;
    |x: String| x;
    |x: String| x;
    |x: String| x;
    String::from("x".as_ref());
    |x: String| x;
    |x: String| x;
    |x: String| x;
    |x: String| x;
}

fn err_first_local_pat() {
    String::from("x".as_ref());
    let _ = String::from("x");
}

fn err_second_local_pat() {
    let _ = String::from("x");
    String::from("x".as_ref());
}

fn err_mid_local_pat() {
    let _ = String::from("x");
    let _ = String::from("x");
    let _ = String::from("x");
    let _ = String::from("x");
    String::from("x".as_ref());
    let _ = String::from("x");
    let _ = String::from("x");
    let _ = String::from("x");
    let _ = String::from("x");
}

fn main() {}
