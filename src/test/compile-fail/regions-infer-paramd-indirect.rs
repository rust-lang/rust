// Check that we correctly infer that b and c must be region
// parameterized because they reference a which requires a region.

type a = &int;
type b = @a;
type c = {f: @b};

impl methods for c {
    fn set_f_ok(b: @b/&self) {
        self.f = b;
    }

    fn set_f_bad(b: @b) {
        self.f = b; //~ ERROR mismatched types: expected `@@&self.int` but found `@@&int`
    }
}

fn main() {}