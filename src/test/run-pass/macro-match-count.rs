// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(macro_rules)];

pub fn main() {
    macro_rules! count_arguments(
        (single $e:expr) => ( #($e) );
        (sequence $($e:expr)*) => ( #($e) );
        (sequence_with_delimiter $($e:expr),*) => ( #($e) );
        (nested_sequence_total_count { $($tk:expr => { $($k:expr => $v:expr),* }),* }) =>
            ( (#($tk), #($k), #($v)) );
    )

    assert_eq!(count_arguments!(single 42), 1);
    assert_eq!(count_arguments!(sequence 1 2 3 4 5), 5);
    assert_eq!(count_arguments!(sequence_with_delimiter 1, 2, 3, 4), 4);
    assert_eq!(count_arguments!(nested_sequence_total_count {
        "key" => {
            0.1 => 0.3,
            0.2 => 0.4
        },
        "value" => {
            5 => true,
            42 => false
        }
    }), (2, 4, 4));
}
