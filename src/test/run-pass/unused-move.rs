// Issue #3878
// Issue Name: Unused move causes a crash
// Abstract: zero-fill to block after drop

// pretty-expanded FIXME #23616

#![allow(path_statements)]
#![feature(box_syntax)]

pub fn main()
{
    let y: Box<_> = box 1;
    y;
}
