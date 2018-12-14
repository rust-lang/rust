// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::all, clippy::similar_names)]
#![allow(unused, clippy::println_empty_string)]

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
    let Foo {
        apple: spring,
        bpple: sprang,
    } = unimplemented!();
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

fn underscores_and_numbers() {
    let _1 = 1; //~ERROR Consider a more descriptive name
    let ____1 = 1; //~ERROR Consider a more descriptive name
    let __1___2 = 12; //~ERROR Consider a more descriptive name
    let _1_ok = 1;
}

fn issue2927() {
    let args = 1;
    format!("{:?}", 2);
}

fn issue3078() {
    match "a" {
        stringify!(a) => {},
        _ => {},
    }
}

struct Bar;

impl Bar {
    fn bar() {
        let _1 = 1;
        let ____1 = 1;
        let __1___2 = 12;
        let _1_ok = 1;
    }
}

// false positive similar_names (#3057, #2651)
// clippy claimed total_reg_src_size and total_size and
// numb_reg_src_checkouts and total_bin_size were similar
#[derive(Debug, Clone)]
pub(crate) struct DirSizes {
    pub(crate) total_size: u64,
    pub(crate) numb_bins: u64,
    pub(crate) total_bin_size: u64,
    pub(crate) total_reg_size: u64,
    pub(crate) total_git_db_size: u64,
    pub(crate) total_git_repos_bare_size: u64,
    pub(crate) numb_git_repos_bare_repos: u64,
    pub(crate) numb_git_checkouts: u64,
    pub(crate) total_git_chk_size: u64,
    pub(crate) total_reg_cache_size: u64,
    pub(crate) total_reg_src_size: u64,
    pub(crate) numb_reg_cache_entries: u64,
    pub(crate) numb_reg_src_checkouts: u64,
}
