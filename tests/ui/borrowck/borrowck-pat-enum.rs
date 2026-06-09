//@ run-pass
#![allow(dead_code)]

fn match_ref(v: Option<isize>) -> isize {
    match v {
      Some(ref i) => {
        *i
      }
      None => {0}
    }
}

fn match_ref_unused(v: Option<isize>) {
    match v {
      Some(_) => {}
      None => {}
    }
}

fn impure(_i: isize) {
}

fn match_imm_reg(v: &Option<isize>) {
    match *v {
      Some(ref i) => {impure(*i)} // OK because immutable
      None => {}
    }
}

fn match_mut_reg(v: &mut Option<isize>) {
    match *v {
      Some(ref i) => {impure(*i)} // OK, frozen
      None => {}
    }
}

pub fn main() {
}
