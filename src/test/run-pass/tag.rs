


// -*- rust -*-
enum colour { red(int, int), green, }

impl colour : cmp::Eq {
    pure fn eq(other: &colour) -> bool {
        match self {
            red(a0, b0) => {
                match (*other) {
                    red(a1, b1) => a0 == a1 && b0 == b1,
                    green => false,
                }
            }
            green => {
                match (*other) {
                    red(*) => false,
                    green => true
                }
            }
        }
    }
    pure fn ne(other: &colour) -> bool { !self.eq(other) }
}

fn f() { let x = red(1, 2); let y = green; assert (x != y); }

fn main() { f(); }
