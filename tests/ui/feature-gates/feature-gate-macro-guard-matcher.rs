fn main() {
    macro_rules! m {
        ($x:guard) => {}; //~ ERROR `guard` fragments in macro are unstable
    }
}
