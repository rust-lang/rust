//@ edition:2024
// ignore-tidy-linelength

macro_rules! demo1 {
    ( $a:tt ) => { println!("one tokens") };
}

macro_rules! demo2 {
    ( $a:tt $b:tt ) => { println!("two tokens") };
}

macro_rules! demo3 {
    ( $a:tt $b:tt $c:tt ) => { println!("three tokens") };
}

macro_rules! demo4 {
    ( $a:tt $b:tt $c:tt $d:tt ) => { println!("four tokens") };
}

macro_rules! demo5 {
    ( $a:tt $b:tt $c:tt $d:tt $e:tt ) => { println!("five tokens") };
}

macro_rules! demo6 {
    ( $a:tt $b:tt $c:tt $d:tt $e:tt $f:tt ) => { println!("six tokens") };
}

macro_rules! demo7 {
    ( $a:tt $b:tt $c:tt $d:tt $e:tt $f:tt $g:tt ) => { println!("seven tokens") };
}

macro_rules! demon {
    ( $($n:tt)* ) => { println!("unknown number of tokens") };
}

fn main() {
    demo1!("");
    demo2!(# "");
    demo3!(# ""#);
    demo2!(# "foo");
    demo3!(# "foo"#);
    demo2!("foo"#);

    demo2!(blah"xx"); //~ ERROR prefix `blah` is unknown
    demo2!(blah#"xx"#);
    //~^ ERROR prefix `blah` is unknown
    //~| ERROR invalid string literal

    demo2!(## "foo"); //~ ERROR reserved multi-hash token is forbidden
    demo3!("foo"###); //~ ERROR reserved multi-hash token is forbidden
    demo3!(### "foo"); //~ ERROR reserved multi-hash token is forbidden
    demo3!(## "foo"#); //~ ERROR reserved multi-hash token is forbidden
    demo5!(### "foo"###);
    //~^ ERROR reserved multi-hash token is forbidden
    //~| ERROR reserved multi-hash token is forbidden

    demo1!(#""); //~ ERROR invalid string literal
    demo1!(#""#); //~ ERROR invalid string literal
    demo1!(####""); //~ ERROR invalid string literal
    demo1!(#"foo"); //~ ERROR invalid string literal
    demo1!(###"foo"); //~ ERROR invalid string literal
    demo1!(#"foo"#); //~ ERROR invalid string literal
    demo1!(###"foo"#); //~ ERROR invalid string literal
    demo1!(###"foo"##); //~ ERROR invalid string literal
    demo1!(###"foo"###); //~ ERROR invalid string literal
    demo2!(#"foo"###);
    //~^ ERROR invalid string literal
    //~| ERROR reserved multi-hash token is forbidden

    // More than 255 hashes
    demon!(####################################################################################################################################################################################################################################################################"foo");
    //~^ ERROR invalid string literal
}
