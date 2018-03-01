fn fn_val(i: i32) -> i32 { unimplemented!() }
fn fn_constref(i: &i32) -> i32 { unimplemented!() }
fn fn_mutref(i: &mut i32) { unimplemented!() }
fn foo() -> i32 { unimplemented!() }

fn immutable_condition() {
    // Should warn when all vars mentionned are immutable
    let y = 0;
    while y < 10 {
        println!("KO - y is immutable");
    }

    let x = 0;
    while y < 10 && x < 3 {
        println!("KO - x and y immutable");
    }

    let cond = false;
    while !cond {
        println!("KO - cond immutable");
    }

    let mut i = 0;
    while y < 10 && i < 3 {
        i += 1;
        println!("OK - i is mutable");
    }

    let mut mut_cond = false;
    while !mut_cond || cond {
        mut_cond = true;
        println!("OK - mut_cond is mutable");
    }

    while foo() < x {
        println!("OK - Fn call results may vary");
    }

}

fn unused_var() {
    // Should warn when a (mutable) var is not used in while body
    let (mut i, mut j) = (0, 0);

    while i < 3 {
        j = 3;
        println!("KO - i not mentionned");
    }

    while i < 3 && j > 0 {
        println!("KO - i and j not mentionned");
    }

    while i < 3 {
        let mut i = 5;
        fn_mutref(&mut i);
        println!("KO - shadowed");
    }

    while i < 3 && j > 0 {
        i = 5;
        println!("OK - i in cond and mentionned");
    }
}

fn used_immutable() {
    let mut i = 0;

    while i < 3 {
        fn_constref(&i);
        println!("KO - const reference");
    }

    while i < 3 {
        fn_val(i);
        println!("KO - passed by value");
    }

    while i < 3 {
        println!("OK - passed by mutable reference");
        fn_mutref(&mut i)
    }
}

fn main() {
    immutable_condition();
    unused_var();
    used_immutable();
}
