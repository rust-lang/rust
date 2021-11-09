#![warn(clippy::similar_names)]
#![allow(
    unused,
    clippy::println_empty_string,
    clippy::empty_loop,
    clippy::diverging_sub_expression
)]

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

    // names often used in win32 code (for example WindowProc)
    let wparam: i32;
    let lparam: i32;

    let iter: i32;
    let item: i32;
}

fn foo() {
    let Foo { apple, bpple } = unimplemented!();
    let Foo {
        apple: spring,
        bpple: sprang,
    } = unimplemented!();
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

fn ignore_underscore_prefix() {
    let hello: ();
    let _hello: ();
}
