// Test for #120601, which causes an ice bug cause of unexpected type
//
//@ parallel-front-end-robustness
//@ compile-flags: -Z threads=40
struct T;
struct Tuple(i32);

async fn foo() -> Result<(), ()> {
    Unstable2(())
}

async fn tuple() -> Tuple {
    Tuple(1i32)
}

async fn match_() {
    match tuple() {
        Tuple(_) => {}
    }
}

fn main() {}
