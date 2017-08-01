#![feature(plugin)]
#![plugin(clippy)]
#![warn(clippy,similar_names)]
#![allow(unused)]


struct Foo {
    apple: i32,
    bpple: i32,
}

fn main() {
    let specter: i32;
    let spectre: i32;

    let apple: i32;

    let bpple: i32;

    let cpple: i32;


    let a_bar: i32;
    let b_bar: i32;
    let c_bar: i32;

    let items = [5];
    for item in &items {
        loop {}
    }

    let foo_x: i32;
    let foo_y: i32;

    let rhs: i32;
    let lhs: i32;

    let bla_rhs: i32;
    let bla_lhs: i32;

    let blubrhs: i32;
    let blublhs: i32;

    let blubx: i32;
    let bluby: i32;


    let cake: i32;
    let cakes: i32;
    let coke: i32;

    match 5 {
        cheese @ 1 => {},
        rabbit => panic!(),
    }
    let cheese: i32;
    match (42, 43) {
        (cheese1, 1) => {},
        (cheese2, 2) => panic!(),
        _ => println!(""),
    }
    let ipv4: i32;
    let ipv6: i32;
    let abcd1: i32;
    let abdc2: i32;
    let xyz1abc: i32;
    let xyz2abc: i32;
    let xyzeabc: i32;

    let parser: i32;
    let parsed: i32;
    let parsee: i32;


    let setter: i32;
    let getter: i32;
    let tx1: i32;
    let rx1: i32;
    let tx_cake: i32;
    let rx_cake: i32;
}

fn foo() {
    let Foo { apple, bpple } = unimplemented!();
    let Foo { apple: spring,
        bpple: sprang } = unimplemented!();
}

#[derive(Clone, Debug)]
enum MaybeInst {
    Split,
    Split1(usize),
    Split2(usize),
}

struct InstSplit {
    uiae: usize,
}

impl MaybeInst {
    fn fill(&mut self) {
        let filled = match *self {
            MaybeInst::Split1(goto1) => panic!(1),
            MaybeInst::Split2(goto2) => panic!(2),
            _ => unimplemented!(),
        };
        unimplemented!()
    }
}

fn bla() {
    let a: i32;
    let (b, c, d): (i32, i64, i16);
    {
        {
            let cdefg: i32;
            let blar: i32;
        }
        {
            let e: i32;
        }
        {
            let e: i32;
            let f: i32;

        }
        match 5 {
            1 => println!(""),
            e => panic!(),
        }
        match 5 {
            1 => println!(""),
            _ => panic!(),
        }
    }
}
