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
    demo3!(foo#bar);   //~ ERROR prefix `foo` is unknown
    demo2!(foo"bar");  //~ ERROR prefix `foo` is unknown
    demo2!(foo'b');    //~ ERROR prefix `foo` is unknown

    demo2!(foo'b);     //~ ERROR prefix `foo` is unknown
    demo3!(foo# bar);  //~ ERROR prefix `foo` is unknown
    demo4!(foo#! bar); //~ ERROR prefix `foo` is unknown
    demo4!(foo## bar); //~ ERROR prefix `foo` is unknown

    demo4!(foo#bar#);
    //~^ ERROR prefix `foo` is unknown
    //~| ERROR prefix `bar` is unknown

    demo3!(foo # bar);
    demo3!(foo #bar);
    demo4!(foo!#bar);
    demo4!(foo ##bar);

    demo3!(r"foo"#bar);
    demo3!(r#foo#bar);
}
