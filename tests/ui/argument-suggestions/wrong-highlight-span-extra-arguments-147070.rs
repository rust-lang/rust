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
    let foo = Thingie::new( //~ ERROR: this function takes 6 arguments but 7 arguments were supplied
        String::from(""),
        String::from(""),
        String::from(""),
        String::from(""),
        String::from(""),
        String::from(""),
        String::from(""),
    );
}
