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
