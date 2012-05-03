// xfail-win32
// Passing enums by value

enum void { }

#[nolink]
native mod bindgen {
    fn printf(++v: void);
}

fn main() { }
