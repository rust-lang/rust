// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern {
    fn write(fildes: i32, buf: *const i8, nbyte: u64) -> i64;
}

#[inline(always)]
fn size_of<T>(_: T) -> usize {
    ::std::mem::size_of::<T>()
}

macro_rules! write {
    ($arr:expr) => {{
        #[allow(non_upper_case_globals)]
        const stdout: i32 = 1;
        unsafe {
            write(stdout, $arr.as_ptr() as *const i8,
                  $arr.len() * size_of($arr[0]));
        }
    }}
}

macro_rules! cast {
    ($x:expr) => ($x as ())
}

fn main() {
    let hello = ['H', 'e', 'y'];
    write!(hello);
    cast!(2);
}
