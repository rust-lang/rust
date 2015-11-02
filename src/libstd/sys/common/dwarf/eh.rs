// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Parsing of GCC-style Language-Specific Data Area (LSDA)
//! For details see:
//!   http://refspecs.linuxfoundation.org/LSB_3.0.0/LSB-PDA/LSB-PDA/ehframechpt.html
//!   http://mentorembedded.github.io/cxx-abi/exceptions.pdf
//!   http://www.airs.com/blog/archives/460
//!   http://www.airs.com/blog/archives/464
//!
//! A reference implementation may be found in the GCC source tree
//! (<root>/libgcc/unwind-c.c as of this writing)

#![allow(non_upper_case_globals)]
#![allow(unused)]

use prelude::v1::*;
use sys::common::dwarf::DwarfReader;
use core::mem;

pub const DW_EH_PE_omit     : u8 = 0xFF;
pub const DW_EH_PE_absptr   : u8 = 0x00;

pub const DW_EH_PE_uleb128  : u8 = 0x01;
pub const DW_EH_PE_udata2   : u8 = 0x02;
pub const DW_EH_PE_udata4   : u8 = 0x03;
pub const DW_EH_PE_udata8   : u8 = 0x04;
pub const DW_EH_PE_sleb128  : u8 = 0x09;
pub const DW_EH_PE_sdata2   : u8 = 0x0A;
pub const DW_EH_PE_sdata4   : u8 = 0x0B;
pub const DW_EH_PE_sdata8   : u8 = 0x0C;

pub const DW_EH_PE_pcrel    : u8 = 0x10;
pub const DW_EH_PE_textrel  : u8 = 0x20;
pub const DW_EH_PE_datarel  : u8 = 0x30;
pub const DW_EH_PE_funcrel  : u8 = 0x40;
pub const DW_EH_PE_aligned  : u8 = 0x50;

pub const DW_EH_PE_indirect : u8 = 0x80;

#[derive(Copy, Clone)]
pub struct EHContext {
    pub ip: usize,         // Current instruction pointer
    pub func_start: usize, // Address of the current function
    pub text_start: usize, // Address of the code section
    pub data_start: usize, // Address of the data section
}

pub unsafe fn find_landing_pad(lsda: *const u8, context: &EHContext)
                               -> Option<usize> {
    if lsda.is_null() {
        return None;
    }

    let func_start = context.func_start;
    let mut reader = DwarfReader::new(lsda);

    let start_encoding = reader.read::<u8>();
    // base address for landing pad offsets
    let lpad_base = if start_encoding != DW_EH_PE_omit {
        read_encoded_pointer(&mut reader, context, start_encoding)
    } else {
        func_start
    };

    let ttype_encoding = reader.read::<u8>();
    if ttype_encoding != DW_EH_PE_omit {
        // Rust doesn't analyze exception types, so we don't care about the type table
        reader.read_uleb128();
    }

    let call_site_encoding = reader.read::<u8>();
    let call_site_table_length = reader.read_uleb128();
    let action_table = reader.ptr.offset(call_site_table_length as isize);
    // Return addresses point 1 byte past the call instruction, which could
    // be in the next IP range.
    let ip = context.ip-1;

    while reader.ptr < action_table {
        let cs_start = read_encoded_pointer(&mut reader, context, call_site_encoding);
        let cs_len = read_encoded_pointer(&mut reader, context, call_site_encoding);
        let cs_lpad = read_encoded_pointer(&mut reader, context, call_site_encoding);
        let cs_action = reader.read_uleb128();
        // Callsite table is sorted by cs_start, so if we've passed the ip, we
        // may stop searching.
        if ip < func_start + cs_start {
            break
        }
        if ip < func_start + cs_start + cs_len {
            if cs_lpad != 0 {
                return Some(lpad_base + cs_lpad);
            } else {
                return None;
            }
        }
    }
    // IP range not found: gcc's C++ personality calls terminate() here,
    // however the rest of the languages treat this the same as cs_lpad == 0.
    // We follow this suit.
    None
}

#[inline]
fn round_up(unrounded: usize, align: usize) -> usize {
    assert!(align.is_power_of_two());
    (unrounded + align - 1) & !(align - 1)
}

unsafe fn read_encoded_pointer(reader: &mut DwarfReader,
                               context: &EHContext,
                               encoding: u8) -> usize {
    assert!(encoding != DW_EH_PE_omit);

    // DW_EH_PE_aligned implies it's an absolute pointer value
    if encoding == DW_EH_PE_aligned {
        reader.ptr = round_up(reader.ptr as usize,
                              mem::size_of::<usize>()) as *const u8;
        return reader.read::<usize>();
    }

    let mut result = match encoding & 0x0F {
        DW_EH_PE_absptr => reader.read::<usize>(),
        DW_EH_PE_uleb128 => reader.read_uleb128() as usize,
        DW_EH_PE_udata2 => reader.read::<u16>() as usize,
        DW_EH_PE_udata4 => reader.read::<u32>() as usize,
        DW_EH_PE_udata8 => reader.read::<u64>() as usize,
        DW_EH_PE_sleb128 => reader.read_sleb128() as usize,
        DW_EH_PE_sdata2 => reader.read::<i16>() as usize,
        DW_EH_PE_sdata4 => reader.read::<i32>() as usize,
        DW_EH_PE_sdata8 => reader.read::<i64>() as usize,
        _ => panic!()
    };

    result += match encoding & 0x70 {
        DW_EH_PE_absptr => 0,
        // relative to address of the encoded value, despite the name
        DW_EH_PE_pcrel => reader.ptr as usize,
        DW_EH_PE_textrel => { assert!(context.text_start != 0);
                              context.text_start },
        DW_EH_PE_datarel => { assert!(context.data_start != 0);
                              context.data_start },
        DW_EH_PE_funcrel => { assert!(context.func_start != 0);
                              context.func_start },
        _ => panic!()
    };

    if encoding & DW_EH_PE_indirect != 0 {
        result = *(result as *const usize);
    }

    result
}
