// compile-flags: -Z parse-only -Z continue-parse-after-error

fn main() {
    struct Test {
        Very
        Bad //~ ERROR found `Bad`
        Stuff
    }
}
