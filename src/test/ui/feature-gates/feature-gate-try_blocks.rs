// compile-flags: --edition 2018

#[cfg(FALSE)]
fn foo() {
    let try_result: Option<_> = try { //~ ERROR `try` blocks are unstable
        let x = 5;
        x
    };
    assert_eq!(try_result, Some(5));
}

fn main() {}
