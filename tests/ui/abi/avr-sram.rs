//@ add-minicore
//@ revisions: has_sram no_sram disable_sram
//@ build-pass
//@[has_sram] compile-flags: --target avr-none -C target-cpu=atmega328p
//@[has_sram] needs-llvm-components: avr
//@[no_sram] compile-flags: --target avr-none -C target-cpu=attiny11
//@[no_sram] needs-llvm-components: avr
//@[disable_sram] compile-flags: --target avr-none -C target-cpu=atmega328p -C target-feature=-sram
//@[disable_sram] needs-llvm-components: avr
//@ ignore-backends: gcc
//[no_sram,disable_sram]~? WARN target feature `sram` must be enabled
//[disable_sram]~? WARN target feature `sram` cannot be disabled with `-Ctarget-feature`

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;
