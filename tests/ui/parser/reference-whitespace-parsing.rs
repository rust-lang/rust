//! Test parsing of multiple references with various whitespace arrangements

//@ run-pass

#![allow(dead_code)]

fn check_expr() {
    let _:         & usize =     &1;
    let _:       & & usize =    &&1;
    let _:     & & & usize =   &&&1;
    let _:     & & & usize =  & &&1;
    let _:   & & & & usize =  &&&&1;
    let _:   & & & & usize = & &&&1;
    let _: & & & & & usize = &&&&&1;
}

fn check_ty() {
    let _:     &usize =         & 1;
    let _:    &&usize =       & & 1;
    let _:   &&&usize =     & & & 1;
    let _:  & &&usize =     & & & 1;
    let _:  &&&&usize =   & & & & 1;
    let _: & &&&usize =   & & & & 1;
    let _: &&&&&usize = & & & & & 1;
}

fn check_pat() {
    let     &_ =         & 1_usize;
    let    &&_ =       & & 1_usize;
    let   &&&_ =     & & & 1_usize;
    let  & &&_ =     & & & 1_usize;
    let  &&&&_ =   & & & & 1_usize;
    let & &&&_ =   & & & & 1_usize;
    let &&&&&_ = & & & & & 1_usize;
}

pub fn main() {}
