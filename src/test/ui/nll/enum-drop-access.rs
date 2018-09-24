#![feature(nll)]

enum DropOption<T> {
    Some(T),
    None,
}

impl<T> Drop for DropOption<T> {
    fn drop(&mut self) {}
}

// Dropping opt could access the value behind the reference,
fn drop_enum(opt: DropOption<&mut i32>) -> Option<&mut i32> {
    match opt {
        DropOption::Some(&mut ref mut r) => { //~ ERROR
            Some(r)
        },
        DropOption::None => None,
    }
}

fn optional_drop_enum(opt: Option<DropOption<&mut i32>>) -> Option<&mut i32> {
    match opt {
        Some(DropOption::Some(&mut ref mut r)) => { //~ ERROR
            Some(r)
        },
        Some(DropOption::None) | None => None,
    }
}

// Ok, dropping opt doesn't access the reference
fn optional_tuple(opt: Option<(&mut i32, String)>) -> Option<&mut i32> {
    match opt {
        Some((&mut ref mut r, _)) => {
            Some(r)
        },
        None => None,
    }
}

// Ok, dropping res doesn't access the Ok case.
fn different_variants(res: Result<&mut i32, String>) -> Option<&mut i32> {
    match res {
        Ok(&mut ref mut r) => {
            Some(r)
        },
        Err(_) => None,
    }
}

fn main() {}
