// xfail-win32
// Passing enums by value

enum void { }

#[nolink]
extern mod bindgen {
    #[legacy_exports];
    fn printf(++v: void);
}

fn main() { }
