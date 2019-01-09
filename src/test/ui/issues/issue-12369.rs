#![feature(slice_patterns)]
#![deny(unreachable_patterns)]

fn main() {
    let sl = vec![1,2,3];
    let v: isize = match &*sl {
        &[] => 0,
        &[a,b,c] => 3,
        &[a, ref rest..] => a,
        &[10,a, ref rest..] => 10 //~ ERROR: unreachable pattern
    };
}
