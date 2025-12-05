// Regression test for #83621.

extern "C" {
    static x: _; //~ ERROR: [E0121]
}

fn main() {}
