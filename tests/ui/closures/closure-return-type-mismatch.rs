fn main() {
    || {
        if false {
            return "test";
        }
        let a = true;
        a //~ ERROR mismatched types
    };

    || -> bool {
        if false {
            return "hello" //~ ERROR mismatched types
        };
        let b = true;
        b
    };
}

// issue: rust-lang/rust#130858 rust-lang/rust#125655
static FOO: fn() -> bool = || -> bool { 1 };
//~^ ERROR mismatched types
