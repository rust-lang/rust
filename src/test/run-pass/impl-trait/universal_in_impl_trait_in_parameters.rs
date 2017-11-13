// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(universal_impl_trait)]
use std::fmt::Display;

fn check_display_eq(iter: impl IntoIterator<Item = impl Display>) {
    let mut collected = String::new();
    for it in iter {
        let disp = format!("{} ", it);
        collected.push_str(&disp);
    }
    assert_eq!("0 3 27 823 4891 1 0", collected.trim());
}

fn main() {
    let i32_list = [0i32, 3, 27, 823, 4891, 1, 0];
    let i32_list_vec = vec![0i32, 3, 27, 823, 4891, 1, 0];
    let u32_list = [0u32, 3, 27, 823, 4891, 1, 0];
    let u32_list_vec = vec![0u32, 3, 27, 823, 4891, 1, 0];
    let u16_list = [0u16, 3, 27, 823, 4891, 1, 0];
    let str_list = ["0", "3", "27", "823", "4891", "1", "0"];
    let str_list_vec = vec!["0", "3", "27", "823", "4891", "1", "0"];

    check_display_eq(&i32_list);
    check_display_eq(i32_list_vec);
    check_display_eq(&u32_list);
    check_display_eq(u32_list_vec);
    check_display_eq(&u16_list);
    check_display_eq(&str_list);
    check_display_eq(str_list_vec);
}
