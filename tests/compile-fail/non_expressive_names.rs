#![feature(plugin)]
#![plugin(clippy)]
#![deny(clippy)]
#![allow(unused)]

fn main() {
    let specter: i32;
    let spectre: i32;

    let apple: i32; //~ NOTE: existing binding defined here
    let bpple: i32; //~ ERROR: name is too similar
    let cpple: i32; //~ ERROR: name is too similar

    let a_bar: i32;
    let b_bar: i32;
    let c_bar: i32;

    let foo_x: i32;
    let foo_y: i32;

    let rhs: i32;
    let lhs: i32;

    let bla_rhs: i32;
    let bla_lhs: i32;

    let blubrhs: i32; //~ NOTE: existing binding defined here
    let blublhs: i32; //~ ERROR: name is too similar

    let blubx: i32; //~ NOTE: existing binding defined here
    let bluby: i32; //~ ERROR: name is too similar
    //~| HELP: separate the discriminating character by an underscore like: `blub_y`

    let cake: i32; //~ NOTE: existing binding defined here
    let cakes: i32;
    let coke: i32; //~ ERROR: name is too similar

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
            let e: i32; //~ ERROR: 5th binding whose name is just one char
        }
        {
            let e: i32; //~ ERROR: 5th binding whose name is just one char
            let f: i32; //~ ERROR: 6th binding whose name is just one char
        }
        match 5 {
            1 => println!(""),
            e => panic!(), //~ ERROR: 5th binding whose name is just one char
        }
        match 5 {
            1 => println!(""),
            _ => panic!(),
        }
    }
}
