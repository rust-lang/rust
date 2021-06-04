#![deny(large_assignments)]
#![feature(large_assignments)]
#![move_size_limit = "1000"]
// build-fail
// only-x86_64

// edition:2018

fn main() {
    let x = async { //~ ERROR large_assignments
        let y = [0; 9999];
        dbg!(y);
        thing(&y).await;
        dbg!(y);
    };
    let z = (x, 42); //~ ERROR large_assignments
    //~^ ERROR large_assignments
    let a = z.0; //~ ERROR large_assignments
    let b = z.1;
}

async fn thing(y: &[u8]) {
    dbg!(y);
}
