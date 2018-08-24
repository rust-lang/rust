static TAB: [&mut [u8]; 0] = [];

pub unsafe fn test() {
    TAB[0].iter_mut(); //~ ERROR cannot borrow data mutably in a `&` reference [E0389]
}

pub fn main() {}
