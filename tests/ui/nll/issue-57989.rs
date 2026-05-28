// Test for ICE from issue 57989

fn f(x: &i32) {
    let g = &x;
    *x = 0;     //~ ERROR cannot assign to `*x`, which is behind a `&` reference
                //~| ERROR cannot assign to `*x` because it is borrowed
    g;
}

fn main() {}
