// run-pass

macro_rules! stringify_item {
    ($item:item) => {
        stringify!($item)
    };
}

macro_rules! repro {
    ($expr:expr) => {
        stringify_item! {
            pub fn repro() -> bool {
                $expr
            }
        }
    };
}

fn main() {
    assert_eq!(
        repro!(match () { () => true } | true),
        // njn: this is incorrect, but hopefully #114571 will fix it
        //"pub fn repro() -> bool { (match () { () => true, }) | true }"
        "pub fn repro() -> bool { match() { () => true } | true }"
    );
}
