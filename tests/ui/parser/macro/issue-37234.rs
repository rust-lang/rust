macro_rules! failed {
    () => {{
        let x = 5 ""; //~ ERROR found `""`
    }}
}

fn main() {
    failed!();
}
