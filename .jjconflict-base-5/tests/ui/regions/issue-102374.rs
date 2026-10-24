use std::cell::Cell;

#[rustfmt::skip]
fn f(
    f: for<'a, 'b, 'c, 'd, 'e, 'f, 'g,
           'h, 'i, 'j, 'k, 'l, 'm, 'n,
           'o, 'p, 'q, 'r, 's, 't, 'u,
           'v, 'w, 'x, 'y, 'z, 'z0>
        fn(Cell<(&   i32, &'a i32, &'b i32, &'c i32, &'d i32,
                 &'e i32, &'f i32, &'g i32, &'h i32, &'i i32,
                 &'j i32, &'k i32, &'l i32, &'m i32, &'n i32,
                 &'o i32, &'p i32, &'q i32, &'r i32, &'s i32,
                 &'t i32, &'u i32, &'v i32, &'w i32, &'x i32,
                 &'y i32, &'z i32, &'z0 i32)>),
) -> i32 {
    f
    //~^ ERROR mismatched types
}

fn main() {}
