// compile-flags: -Z parse-only -Z continue-parse-after-error

fn main() {
    enum Test {
        Very
        Bad //~ ERROR found `Bad`
        Stuff
    }
}
