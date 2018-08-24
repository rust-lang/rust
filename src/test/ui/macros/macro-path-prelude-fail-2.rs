mod m {
    fn check() {
        Result::Ok!(); //~ ERROR fail to resolve non-ident macro path
    }
}

fn main() {}
