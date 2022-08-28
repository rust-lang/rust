//! Parsing of GCC-style Language-Specific Data Area (LSDA)
//! For details see:
//!  * <https://refspecs.linuxfoundation.org/LSB_3.0.0/LSB-PDA/LSB-PDA/ehframechpt.html>
//!  * <https://itanium-cxx-abi.github.io/cxx-abi/exceptions.pdf>
//!  * <https://www.airs.com/blog/archives/460>
//!  * <https://www.airs.com/blog/archives/464>
//!
//! A reference implementation may be found in the GCC source tree
//! (`<root>/libgcc/unwind-c.c` as of this writing).

#![allow(non_upper_case_globals)]
#![allow(unused)]

use super::DwarfReader;
use core::mem;

pub const DW_EH_PE_omit: u8 = 0xFF;
pub const DW_EH_PE_absptr: u8 = 0x00;

pub const DW_EH_PE_uleb128: u8 = 0x01;
pub const DW_EH_PE_udata2: u8 = 0x02;
pub const DW_EH_PE_udata4: u8 = 0x03;
pub const DW_EH_PE_udata8: u8 = 0x04;
pub const DW_EH_PE_sleb128: u8 = 0x09;
pub const DW_EH_PE_sdata2: u8 = 0x0A;
pub const DW_EH_PE_sdata4: u8 = 0x0B;
pub const DW_EH_PE_sdata8: u8 = 0x0C;

pub const DW_EH_PE_pcrel: u8 = 0x10;
pub const DW_EH_PE_textrel: u8 = 0x20;
pub const DW_EH_PE_datarel: u8 = 0x30;
pub const DW_EH_PE_funcrel: u8 = 0x40;
pub const DW_EH_PE_aligned: u8 = 0x50;

pub const DW_EH_PE_indirect: u8 = 0x80;

#[derive(Copy, Clone)]
pub struct EHContext<'a> {
    pub ip: usize,                             // Current instruction pointer
    pub func_start: usize,                     // Address of the current function
    pub get_text_start: &'a dyn Fn() -> usize, // Get address of the code section
    pub get_data_start: &'a dyn Fn() -> usize, // Get address of the data section
}

pub enum EHAction {
    None,
    Cleanup(usize),
    Catch(usize),
    Terminate,
}

pub const USING_SJLJ_EXCEPTIONS: bool = cfg!(all(target_os = "ios", target_arch = "arm"));

pub unsafe fn find_eh_action(lsda: *const u8, context: &EHContext<'_>) -> Result<EHAction, ()> {
    if lsda.is_null() {
        return Ok(EHAction::None);
    }

    let func_start = context.func_start;
    let mut reader = DwarfReader::new(lsda);

    let start_encoding = reader.read::<u8>();
    // base address for landing pad offsets
    let lpad_base = if start_encoding != DW_EH_PE_omit {
        read_encoded_pointer(&mut reader, context, start_encoding)?
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
    let action_table = reader.ptr.add(call_site_table_length as usize);
    let ip = context.ip;

    if !USING_SJLJ_EXCEPTIONS {
        while reader.ptr < action_table {
            let cs_start = read_encoded_pointer(&mut reader, context, call_site_encoding)?;
            let cs_len = read_encoded_pointer(&mut reader, context, call_site_encoding)?;
            let cs_lpad = read_encoded_pointer(&mut reader, context, call_site_encoding)?;
            let cs_action = reader.read_uleb128();
            // Callsite table is sorted by cs_start, so if we've passed the ip, we
            // may stop searching.
            if ip < func_start + cs_start {
                break;
            }
            if ip < func_start + cs_start + cs_len {
                if cs_lpad == 0 {
                    return Ok(EHAction::None);
                } else {
                    let lpad = lpad_base + cs_lpad;
                    return Ok(interpret_cs_action(cs_action, lpad));
                }
            }
        }
        // Ip is not present in the table.  This should not happen... but it does: issue #35011.
        // So rather than returning EHAction::Terminate, we do this.
        Ok(EHAction::None)
    } else {
        // SjLj version:
        // The "IP" is an index into the call-site table, with two exceptions:
        // -1 means 'no-action', and 0 means 'terminate'.
        match ip as isize {
            -1 => return Ok(EHAction::None),
            0 => return Ok(EHAction::Terminate),
            _ => (),
        }
        let mut idx = ip;
        loop {
            let cs_lpad = reader.read_uleb128();
            let cs_action = reader.read_uleb128();
            idx -= 1;
            if idx == 0 {
                // Can never have null landing pad for sjlj -- that would have
                // been indicated by a -1 call site index.
                let lpad = (cs_lpad + 1) as usize;
                return Ok(interpret_cs_action(cs_action, lpad));
            }
        }
    }
}

fn interpret_cs_action(cs_action: u64, lpad: usize) -> EHAction {
    if cs_action == 0 {
        // If cs_action is 0 then this is a cleanup (Drop::drop). We run these
        // for both Rust panics and foreign exceptions.
        EHAction::Cleanup(lpad)
    } else {
        // Stop unwinding Rust panics at catch_unwind.
        EHAction::Catch(lpad)
    }
}

#[inline]
fn round_up(unrounded: usize, align: usize) -> Result<usize, ()> {
    if align.is_power_of_two() { Ok((unrounded + align - 1) & !(align - 1)) } else { Err(()) }
}

unsafe fn read_encoded_pointer(
    reader: &mut DwarfReader,
    context: &EHContext<'_>,
    encoding: u8,
) -> Result<usize, ()> {
    if encoding == DW_EH_PE_omit {
        return Err(());
    }

    // DW_EH_PE_aligned implies it's an absolute pointer value
    if encoding == DW_EH_PE_aligned {
        reader.ptr = round_up(reader.ptr as usize, mem::size_of::<usize>())? as *const u8;
        return Ok(reader.read::<usize>());
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
        _ => return Err(()),
    };

    result += match encoding & 0x70 {
        DW_EH_PE_absptr => 0,
        // relative to address of the encoded value, despite the name
        DW_EH_PE_pcrel => reader.ptr as usize,
        DW_EH_PE_funcrel => {
            if context.func_start == 0 {
                return Err(());
            }
            context.func_start
        }
        DW_EH_PE_textrel => (*context.get_text_start)(),
        DW_EH_PE_datarel => (*context.get_data_start)(),
        _ => return Err(()),
    };

    if encoding & DW_EH_PE_indirect != 0 {
        result = *(result as *const usize);
    }

    Ok(result)
}
