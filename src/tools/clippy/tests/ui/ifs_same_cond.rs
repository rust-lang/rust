#![warn(clippy::ifs_same_cond)]
#![allow(clippy::if_same_then_else, clippy::needless_if, clippy::needless_else)] // all empty blocks

fn ifs_same_cond() {
    let a = 0;
    let b = false;

    if b {
        //~^ ifs_same_cond
    } else if b {
    }

    if b {
        //~^ ifs_same_cond
    } else if b {
    } else if b {
    }

    if a == 1 {
        //~^ ifs_same_cond
    } else if a == 1 {
    }

    if 2 * a == 1 {
        //~^ ifs_same_cond
    } else if 2 * a == 2 {
    } else if 2 * a == 1 {
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

    if let Some(env1) = option_env!("ENV1") {
    } else if let Some(env2) = option_env!("ENV2") {
    }
}

fn issue10272() {
    let a = String::from("ha");
    if a.contains("ah") {
        //~^ ifs_same_cond
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

    let x = std::cell::Cell::new(true);
    if x.get() {
    } else if !x.take() {
    } else if x.get() {
        // ok, x is interior mutable type
    } else {
    }
}

fn main() {}
