//@ run-pass
//@ edition:2021

pub fn foo() {
    let ref_x_ck = 123;
    let _y = || match ref_x_ck {
        2_000_000..=3_999_999 => { println!("A")}
        _ => { println!("B")}
    };
}

fn main() {
    foo();
}
