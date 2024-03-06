//@ run-rustfix
fn main() {
    let _ = vec![1, 2, 3].into_iter().map(|x|
        let y = x; //~ ERROR expected expression, found `let` statement
        y
    );
    let _: () = foo; //~ ERROR mismatched types
}

fn foo() {}
