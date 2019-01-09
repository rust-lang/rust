// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn main() {
    let x = 1;
    let y = &mut x; //[ast]~ ERROR [E0596]
                    //[mir]~^ ERROR [E0596]
}
