// Regression test for https://github.com/rust-lang/rust/issues/166472

fn f(v: &Vec<u8>) {
    ({ *v }).rev();
    //~^ ERROR no method named `rev` found for struct `Vec<u8>` in the current scope
}

fn main() {}
