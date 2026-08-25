mod m {
    fn check() {
        Vec::clone!(); //~ ERROR cannot find
        //~^ NOTE `Vec` is a struct, not a module
        u8::clone!(); //~ ERROR cannot find
        //~^ NOTE `u8` is a builtin type, not a module
    }
}

fn main() {}
