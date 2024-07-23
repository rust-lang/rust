mod m {
    fn check() {
        Result::Ok!(); //~ ERROR cannot find macro `Ok`
    }
}

fn main() {}
