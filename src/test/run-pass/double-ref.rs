#![allow(dead_code)]
// pretty-expanded FIXME #23616

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
