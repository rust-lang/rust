#![feature(if_while_or_patterns)]

fn main() {
    if let 0 | 1 = 0 {
        println!("hello, world");
    };

    if let aaaaaaaaaaaaaaaaaaaaaaaaaa | bbbbbbbbbbbbbbbbbbbbbbbbbbb | cccccccccccccccc | d_100 = 0 {
        println!("hello, world");
    }

    if let aaaaaaaaaaaaaaaaaaaaaaaaaa | bbbbbbbbbbbbbbbbbbbbbbb | ccccccccccccccccccccc | d_101 = 0
    {
        println!("hello, world");
    }

    if let aaaaaaaaaaaaaaaaaaaaaaaaaaaa | bbbbbbbbbbbbbbbbbbbbbbb | ccccccccccccccccccccc | d_103 =
        0
    {
        println!("hello, world");
    }

    if let aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    | bbbbbbbbbbbbbbbbbbbbbbb
    | ccccccccccccccccccccc
    | d_105 = 0
    {
        println!("hello, world");
    }

    while let xxx | xxx | xxx | xxx | xxx | xxx | xxx | xxx | xxx | xxx | xxx | xxx | xxx | xxx
    | xxx | xxx | xxx | xxx | xxx | xxx | xxx | xxx | xxx | xxx | xxx | xxx | xxx = foo_bar(
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
        cccccccccccccccccccccccccccccccccccccccc,
    ) {
        println!("hello, world");
    }
}
