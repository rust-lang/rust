macro_rules! statement {
    () => {;}; //~ ERROR expected expression
}

fn main() {
    let _ = statement!();
}
