// check-compile

struct DisplayOnly;

impl std::fmt::Display for DisplayOnly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

fn main() {
    let x = Some(1);
    println!("{x:?} {x} {x:?}");
    //~^ ERROR: `Option<{integer}>` doesn't implement `std::fmt::Display`
    println!("{x:?} {x} {x:?}", x = Some(1));
    //~^ ERROR: `Option<{integer}>` doesn't implement `std::fmt::Display`
    let x = DisplayOnly;
    println!("{x} {x:?} {x}");
    //~^ ERROR: `DisplayOnly` doesn't implement `Debug`
    println!("{x} {x:?} {x}", x = DisplayOnly);
    //~^ ERROR: `DisplayOnly` doesn't implement `Debug`
}
