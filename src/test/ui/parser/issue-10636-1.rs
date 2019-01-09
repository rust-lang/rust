struct Obj {
    //~^ NOTE: un-closed delimiter
    member: usize
)
//~^ ERROR incorrect close delimiter
//~| NOTE incorrect close delimiter

fn main() {}
