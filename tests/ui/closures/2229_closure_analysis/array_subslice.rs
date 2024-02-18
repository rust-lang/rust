// regression test for #109298
//@ edition: 2021

pub fn subslice_array(x: [u8; 3]) {
    let f = || {
        let [_x @ ..] = x;
        let [ref y, ref mut z @ ..] = x; //~ ERROR cannot borrow `x[..]` as mutable
    };

    f(); //~ ERROR cannot borrow `f` as mutable
}

fn main() {}
