//! regression test for <https://github.com/rust-lang/rust/issues/148467>
//! Ensure the diagnostic suggests `&(mut x)` (parenthesized) instead of `&mut x`.

fn for_loop() {
    let nums: &[u32] = &[1, 2, 3];
    for &num in nums {
        num *= 2; //~ ERROR cannot assign twice to immutable variable `num`
        println!("{num}");
    }
}

fn let_deref(num_ref: &u32) -> u32 {
    let &num = num_ref;

    num *= 2; //~ ERROR cannot assign twice to immutable variable `num`

    num
}

fn deref_inside_pattern(option_num_ref: Option<&u32>) {
    if let Some(&num) = option_num_ref {
        num *= 2; //~ ERROR cannot assign twice to immutable variable `num`

        println!("{num}");
    }
}

/// Insides of deref pattern do not need additional parens
fn inside_of_deref(num_option_ref: &Option<u32>) {
    if let &Some(num) = num_option_ref {
        num *= 2; //~ ERROR cannot assign twice to immutable variable `num`

        println!("{num}");
    }
}

/// &mut deref pattern does not need additional parens
fn let_mut_deref(num_mut_ref: &mut u32) -> u32 {
    let &mut num = num_mut_ref;

    num *= 2; //~ ERROR cannot assign twice to immutable variable `num`

    num
}


fn main() {}
