//@ only-linux
//@ compile-flags: --error-format=human --color=always

// The hightlight span should be correct. See #147070
struct Thingie;

impl Thingie {
    pub(crate) fn new(
        _a: String,
        _b: String,
        _c: String,
        _d: String,
        _e: String,
        _f: String,
    ) -> Self {
        unimplemented!()
    }
}

fn main() {
    let foo = Thingie::new(
        String::from(""),
        String::from(""),
        String::from(""),
        String::from(""),
        String::from(""),
        String::from(""),
        String::from(""),
    );
}
