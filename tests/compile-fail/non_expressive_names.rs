#![feature(plugin)]
#![plugin(clippy)]
#![deny(clippy,similar_names)]
//~^ NOTE: lint level defined here
//~| NOTE: lint level defined here
//~| NOTE: lint level defined here
//~| NOTE: lint level defined here
//~| NOTE: lint level defined here
//~| NOTE: lint level defined here
//~| NOTE: lint level defined here
//~| NOTE: lint level defined here
//~| NOTE: lint level defined here
//~| NOTE: lint level defined here
#![allow(unused)]

fn main() {
    let specter: i32;
    let spectre: i32;

    let apple: i32; //~ NOTE: existing binding defined here
    //~^ NOTE: existing binding defined here
    let bpple: i32; //~ ERROR: name is too similar
    //~| HELP: separate the discriminating character by an underscore like: `b_pple`
    //~| HELP: for further information visit
    let cpple: i32; //~ ERROR: name is too similar
    //~| HELP: separate the discriminating character by an underscore like: `c_pple`
    //~| HELP: for further information visit

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

    let blubx: i32; //~ NOTE: existing binding defined here
    let bluby: i32; //~ ERROR: name is too similar
    //~| HELP: for further information visit
    //~| HELP: separate the discriminating character by an underscore like: `blub_y`

    let cake: i32; //~ NOTE: existing binding defined here
    let cakes: i32;
    let coke: i32; //~ ERROR: name is too similar
    //~| HELP: for further information visit

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
    let xyz1abc: i32; //~ NOTE: existing binding defined here
    let xyz2abc: i32;
    let xyzeabc: i32; //~ ERROR: name is too similar
    //~| HELP: for further information visit

    let parser: i32; //~ NOTE: existing binding defined here
    let parsed: i32;
    let parsee: i32; //~ ERROR: name is too similar
    //~| HELP: for further information visit
    //~| HELP: separate the discriminating character by an underscore like: `parse_e`

    let setter: i32;
    let getter: i32;
    let tx1: i32;
    let rx1: i32;
    let tx_cake: i32;
    let rx_cake: i32;
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
            //~| HELP: for further information visit
        }
        {
            let e: i32; //~ ERROR: 5th binding whose name is just one char
            //~| HELP: for further information visit
            let f: i32; //~ ERROR: 6th binding whose name is just one char
            //~| HELP: for further information visit
        }
        match 5 {
            1 => println!(""),
            e => panic!(), //~ ERROR: 5th binding whose name is just one char
            //~| HELP: for further information visit
        }
        match 5 {
            1 => println!(""),
            _ => panic!(),
        }
    }
}
