//@ edition:2021

macro_rules! demo2 {
    ( $a:tt $b:tt ) => { println!("two tokens") };
}

macro_rules! demo3 {
    ( $a:tt $b:tt $c:tt ) => { println!("three tokens") };
}

macro_rules! demo4 {
    ( $a:tt $b:tt $c:tt $d:tt ) => { println!("four tokens") };
}

fn main() {
    // Non-ascii identifiers
    demo2!(Ñ"foo"); //~ ERROR prefix `Ñ` is unknown
    demo4!(Ñ#""#);  //~ ERROR prefix `Ñ` is unknown
    demo3!(🙃#"");
    //~^ ERROR identifiers cannot contain emoji
}
