mod m {
    fn check() {
        Vec::clone!(); //~ ERROR cannot find module `Vec`
        u8::clone!(); //~ ERROR cannot find module `u8`
    }
}

fn main() {}
