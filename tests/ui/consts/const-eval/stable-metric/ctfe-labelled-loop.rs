// check-fail
// compile-flags: -Z tiny-const-eval-limit

const fn labelled_loop(n: u32) -> u32 {
    let mut i = 0;
    'mylabel: loop { //~ ERROR evaluation of constant value failed [E0080]
        if i > n {
            break 'mylabel
        }
        i += 1;
    }
    0
}

const X: u32 = labelled_loop(19);

fn main() {
    println!("{X}");
}
