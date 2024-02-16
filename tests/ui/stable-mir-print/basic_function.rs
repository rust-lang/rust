//@ compile-flags: -Z unpretty=stable-mir -Z mir-opt-level=3
//@ check-pass
//@ only-x86_64

fn foo(i:i32) -> i32 {
    i + 1
}

fn bar(vec: &mut Vec<i32>) -> Vec<i32> {
    let mut new_vec = vec.clone();
    new_vec.push(1);
    new_vec
}

fn main(){}
