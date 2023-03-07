mod m {
    fn check() {
        Result::Ok!(); //~ ERROR failed to resolve: partially resolved path in a macro
    }
}

fn main() {}
