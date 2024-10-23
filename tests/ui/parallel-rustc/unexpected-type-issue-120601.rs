//@ parallel-front-end
//@ compile-flags: -Z threads=40
struct T;
struct Tuple(i32);

async fn foo() -> Result<(), ()> {
    //~^ ERROR
    Unstable2(())
    //~^ ERROR
}

async fn tuple() -> Tuple {
    //~^ ERROR
    Tuple(1i32)
}

async fn match_() {
    //~^ ERROR
    match tuple() {
        Tuple(_) => {}
        //~^ ERROR
    }
}

fn main() {}
