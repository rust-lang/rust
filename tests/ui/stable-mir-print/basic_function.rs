// compile-flags: -Z unpretty=stable-mir -Z mir-opt-level=3
// check-pass
<<<<<<< HEAD
// only-x86_64
=======
>>>>>>> 9a9a3a91e41 (add stable_mir output test)

fn foo(i:i32) -> i32 {
    i + 1
}

fn bar(vec: &mut Vec<i32>) -> Vec<i32> {
    let mut new_vec = vec.clone();
    new_vec.push(1);
    new_vec
}

fn main(){}
