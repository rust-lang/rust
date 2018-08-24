// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn test() {
    let w: &mut [isize];
    w[5] = 0; //[ast]~ ERROR use of possibly uninitialized variable: `*w` [E0381]
              //[mir]~^ ERROR [E0381]

    let mut w: &mut [isize];
    w[5] = 0; //[ast]~ ERROR use of possibly uninitialized variable: `*w` [E0381]
              //[mir]~^ ERROR [E0381]
}

fn main() { test(); }
