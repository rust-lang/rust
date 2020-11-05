fn main() {
    "Hello, world!".to_string().println!(); //~ ERROR forbidden postfix macro

    "Hello, world!".println!(); //~ ERROR forbidden postfix macro

    false.assert!(); //~ ERROR forbidden postfix macro

    Some(42).assert_eq!(None); //~ ERROR forbidden postfix macro

    std::iter::once(42) //~ ERROR forbidden postfix macro
    //~^ ERROR forbidden postfix macro
        .map(|v| v + 3)
        .dbg!()
        .max()
        .unwrap()
        .dbg!();
}
