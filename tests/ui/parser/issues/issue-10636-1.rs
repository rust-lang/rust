struct Obj {
    //~^ NOTE: unclosed delimiter
    member: usize
)
//~^ ERROR mismatched closing delimiter
//~| NOTE mismatched closing delimiter
//~| NOTE missing open `(` for this delimiter

fn main() {}
