macro_rules! test { ($a, $b) => (()); } //~ ERROR missing fragment

fn main() {
    test!()
}
