struct Obj {
    //~^ NOTE: unclosed delimiter
    member: usize
)
//~^ ERROR mismatched closing delimiter
//~| NOTE mismatched closing delimiter, may missing open `(`

fn main() {}
