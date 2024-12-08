fn main() {
    fn test() {
        loop {
            let blah: Option<String>;
            if true {
                blah = Some("".to_string());
            }
            if let Some(blah) = blah.as_ref() { //~ ERROR E0381
            }
        }
    }
    println!("{:?}", test())
}
