#![warn(clippy::ifs_same_cond)]
#![allow(clippy::if_same_then_else, clippy::comparison_chain)] // all empty blocks

fn ifs_same_cond() {
    let a = 0;
    let b = false;

    if b {
    } else if b {
        //~ ERROR ifs same condition
    }

    if a == 1 {
    } else if a == 1 {
        //~ ERROR ifs same condition
    }

    if 2 * a == 1 {
    } else if 2 * a == 2 {
    } else if 2 * a == 1 {
        //~ ERROR ifs same condition
    } else if a == 1 {
    }

    // See #659
    if cfg!(feature = "feature1-659") {
        1
    } else if cfg!(feature = "feature2-659") {
        2
    } else {
        3
    };

    let mut v = vec![1];
    if v.pop().is_none() {
        // ok, functions
    } else if v.pop().is_none() {
    }

    if v.len() == 42 {
        // ok, functions
    } else if v.len() == 42 {
    }
}

fn issue10272() {
    let a = String::from("ha");
    if a.contains("ah") {
    } else if a.contains("ah") {
        // Trigger this lint
    } else if a.contains("ha") {
    } else if a == "wow" {
    }

    let p: *mut i8 = std::ptr::null_mut();
    if p.is_null() {
    } else if p.align_offset(0) == 0 {
    } else if p.is_null() {
        // ok, p is mutable pointer
    } else {
    }
}

fn main() {}
