struct NonCopy(u8);

const C: () = {
    let &moved = &NonCopy(0);
    //~^ ERROR cannot move out of a shared reference
};

fn main() { }
