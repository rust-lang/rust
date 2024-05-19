enum Fine {
    A(()),
    B([u8; 500]),
}
enum Bad {
    //~^ ERROR: large size difference between variants
    A(()),
    B([u8; 501]),
}

fn main() {}
