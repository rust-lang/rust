fn main() {
    let y = {
        let mut z = 0;
        &mut z
    };
    //~^^ ERROR `z` does not live long enough [E0597]
    println!("{}", y);
}

// https://github.com/rust-lang/rust/issues/464711
