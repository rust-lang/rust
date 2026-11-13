#![allow(clippy::disallowed_names)]

fn main() {
    println!("testing 123");
}

#[cfg(test)]
mod tests {
    fn foo() -> Option<u32> {
        Some(1)
    }

    #[test]
    fn integration_test() {
        // should not lint in test file
        // see https://github.com/rust-lang/rust-clippy/issues/13981
        let bar = foo().unwrap();
        println!("bar: {bar}");
    }
}
