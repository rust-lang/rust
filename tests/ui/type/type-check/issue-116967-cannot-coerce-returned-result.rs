fn foo() -> Result<String, ()> {
    let out: Result<(), ()> = Ok(());
    out //~ ERROR mismatched types
}

fn main() {}
