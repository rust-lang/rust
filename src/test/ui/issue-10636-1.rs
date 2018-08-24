// compile-flags: -Z parse-only

struct Obj { //~ NOTE: unclosed delimiter
    member: usize
)
//~^ ERROR incorrect close delimiter
//~| NOTE incorrect close delimiter

fn main() {}
