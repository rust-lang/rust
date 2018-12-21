// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {Intrinsic, Type};
use IntrinsicDef::Named;

pub fn find(name: &str) -> Option<Intrinsic> {
    intrinsics! {
        name, "x86",
        "_rdrand16_step" => Intrinsic {
            inputs: &[],
            output: &Type::Aggregate(false, &[&::U16, &::I32]),
            definition: Named("llvm.x86.rdrand.16")
        },
        "_rdrand32_step" => Intrinsic {
            inputs: &[],
            output: &Type::Aggregate(false, &[&::U32, &::I32]),
            definition: Named("llvm.x86.rdrand.32")
        },
        "_rdrand64_step" => Intrinsic {
            inputs: &[],
            output: &Type::Aggregate(false, &[&::U64, &::I32]),
            definition: Named("llvm.x86.rdrand.64")
        },
        "_rdseed16_step" => Intrinsic {
            inputs: &[],
            output: &Type::Aggregate(false, &[&::U16, &::I32]),
            definition: Named("llvm.x86.rdseed.16")
        },
        "_rdseed32_step" => Intrinsic {
            inputs: &[],
            output: &Type::Aggregate(false, &[&::U32, &::I32]),
            definition: Named("llvm.x86.rdseed.32")
        },
        "_rdseed64_step" => Intrinsic {
            inputs: &[],
            output: &Type::Aggregate(false, &[&::U64, &::I32]),
            definition: Named("llvm.x86.rdseed.64")
        },
    }
}
