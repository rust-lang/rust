static TAB: [&mut [u8]; 0] = [];

pub unsafe fn test() {
    TAB[0].iter_mut();
    //~^ ERROR cannot borrow `*TAB[_]` as mutable, as `TAB` is an immutable static item [E0596]
}

pub fn main() {}
