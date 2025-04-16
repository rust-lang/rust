//! Parsing of GCC-style Language-Specific Data Area (LSDA)
//! For details see:
//!  * <https://refspecs.linuxfoundation.org/LSB_3.0.0/LSB-PDA/LSB-PDA/ehframechpt.html>
//!  * <https://refspecs.linuxfoundation.org/LSB_5.0.0/LSB-Core-generic/LSB-Core-generic/dwarfext.html>
//!  * <https://itanium-cxx-abi.github.io/cxx-abi/exceptions.pdf>
//!  * <https://www.airs.com/blog/archives/460>
//!  * <https://www.airs.com/blog/archives/464>
//!
//! A reference implementation may be found in the GCC source tree
//! (`<root>/libgcc/unwind-c.c` as of this writing).

#![allow(non_upper_case_globals)]
#![allow(unused)]

use core::ptr;

use super::DwarfReader;

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
    pub ip: *const u8,                             // Current instruction pointer
    pub func_start: *const u8,                     // Pointer to the current function
    pub get_text_start: &'a dyn Fn() -> *const u8, // Get pointer to the code section
    pub get_data_start: &'a dyn Fn() -> *const u8, // Get pointer to the data section
}

/// Landing pad.
type LPad = *const u8;
pub enum EHAction {
    None,
    Cleanup(LPad),
    Catch(LPad),
    Filter(LPad),
    Terminate,
}

/// 32-bit ARM Darwin platforms uses SjLj exceptions.
///
/// The exception is watchOS armv7k (specifically that subarchitecture), which
/// instead uses DWARF Call Frame Information (CFI) unwinding.
///
/// <https://github.com/llvm/llvm-project/blob/llvmorg-18.1.4/clang/lib/Driver/ToolChains/Darwin.cpp#L3107-L3119>
pub const USING_SJLJ_EXCEPTIONS: bool =
    cfg!(all(target_vendor = "apple", not(target_os = "watchos"), target_arch = "arm"));

pub unsafe fn find_eh_action(lsda: *const u8, context: &EHContext<'_>) -> Result<EHAction, ()> {
    if lsda.is_null() {
        return Ok(EHAction::None);
    }

    let func_start = context.func_start;
    let mut reader = DwarfReader::new(lsda);
    let lpad_base = unsafe {
        let start_encoding = reader.read::<u8>();
        // base address for landing pad offsets
        if start_encoding != DW_EH_PE_omit {
            read_encoded_pointer(&mut reader, context, start_encoding)?
        } else {
            func_start
        }
    };
    let call_site_encoding = unsafe {
        let ttype_encoding = reader.read::<u8>();
        if ttype_encoding != DW_EH_PE_omit {
            // Rust doesn't analyze exception types, so we don't care about the type table
            reader.read_uleb128();
        }

        reader.read::<u8>()
    };
    let action_table = unsafe {
        let call_site_table_length = reader.read_uleb128();
        reader.ptr.add(call_site_table_length as usize)
    };
    let ip = context.ip;

    if !USING_SJLJ_EXCEPTIONS {
        // read the callsite table
        while reader.ptr < action_table {
            unsafe {
                // these are offsets rather than pointers;
                let cs_start = read_encoded_offset(&mut reader, call_site_encoding)?;
                let cs_len = read_encoded_offset(&mut reader, call_site_encoding)?;
                let cs_lpad = read_encoded_offset(&mut reader, call_site_encoding)?;
                let cs_action_entry = reader.read_uleb128();
                // Callsite table is sorted by cs_start, so if we've passed the ip, we
                // may stop searching.
                if ip < func_start.wrapping_add(cs_start) {
                    break;
                }
                if ip < func_start.wrapping_add(cs_start + cs_len) {
                    if cs_lpad == 0 {
                        return Ok(EHAction::None);
                    } else {
                        let lpad = lpad_base.wrapping_add(cs_lpad);
                        return Ok(interpret_cs_action(action_table, cs_action_entry, lpad));
                    }
                }
            }
        }
        // Ip is not present in the table. This indicates a nounwind call.
        Ok(EHAction::Terminate)
    } else {
        // SjLj version:
        // The "IP" is an index into the call-site table, with two exceptions:
        // -1 means 'no-action', and 0 means 'terminate'.
        match ip.addr() as isize {
            -1 => return Ok(EHAction::None),
            0 => return Ok(EHAction::Terminate),
            _ => (),
        }
        let mut idx = ip.addr();
        loop {
            let cs_lpad = unsafe { reader.read_uleb128() };
            let cs_action_entry = unsafe { reader.read_uleb128() };
            idx -= 1;
            if idx == 0 {
                // Can never have null landing pad for sjlj -- that would have
                // been indicated by a -1 call site index.
                // FIXME(strict provenance)
                let lpad = ptr::with_exposed_provenance((cs_lpad + 1) as usize);
                return Ok(unsafe { interpret_cs_action(action_table, cs_action_entry, lpad) });
            }
        }
    }
}

unsafe fn interpret_cs_action(
    action_table: *const u8,
    cs_action_entry: u64,
    lpad: LPad,
) -> EHAction {
    if cs_action_entry == 0 {
        // If cs_action_entry is 0 then this is a cleanup (Drop::drop). We run these
        // for both Rust panics and foreign exceptions.
        EHAction::Cleanup(lpad)
    } else {
        // If lpad != 0 and cs_action_entry != 0, we have to check ttype_index.
        // If ttype_index == 0 under the condition, we take cleanup action.
        let action_record = unsafe { action_table.offset(cs_action_entry as isize - 1) };
        let mut action_reader = DwarfReader::new(action_record);
        let ttype_index = unsafe { action_reader.read_sleb128() };
        if ttype_index == 0 {
            EHAction::Cleanup(lpad)
        } else if ttype_index > 0 {
            // Stop unwinding Rust panics at catch_unwind.
            EHAction::Catch(lpad)
        } else {
            EHAction::Filter(lpad)
        }
    }
}

#[inline]
fn round_up(unrounded: usize, align: usize) -> Result<usize, ()> {
    if align.is_power_of_two() { Ok((unrounded + align - 1) & !(align - 1)) } else { Err(()) }
}

/// Reads an offset (`usize`) from `reader` whose encoding is described by `encoding`.
///
/// `encoding` must be a [DWARF Exception Header Encoding as described by the LSB spec][LSB-dwarf-ext].
/// In addition the upper ("application") part must be zero.
///
/// # Errors
/// Returns `Err` if `encoding`
/// * is not a valid DWARF Exception Header Encoding,
/// * is `DW_EH_PE_omit`, or
/// * has a non-zero application part.
///
/// [LSB-dwarf-ext]: https://refspecs.linuxfoundation.org/LSB_5.0.0/LSB-Core-generic/LSB-Core-generic/dwarfext.html
unsafe fn read_encoded_offset(reader: &mut DwarfReader, encoding: u8) -> Result<usize, ()> {
    if encoding == DW_EH_PE_omit || encoding & 0xF0 != 0 {
        return Err(());
    }
    let result = unsafe {
        match encoding & 0x0F {
            // despite the name, LLVM also uses absptr for offsets instead of pointers
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
        }
    };
    Ok(result)
}

/// Reads a pointer from `reader` whose encoding is described by `encoding`.
///
/// `encoding` must be a [DWARF Exception Header Encoding as described by the LSB spec][LSB-dwarf-ext].
///
/// # Errors
/// Returns `Err` if `encoding`
/// * is not a valid DWARF Exception Header Encoding,
/// * is `DW_EH_PE_omit`, or
/// * combines `DW_EH_PE_absptr` or `DW_EH_PE_aligned` application part with an integer encoding
///   (not `DW_EH_PE_absptr`) in the value format part.
///
/// [LSB-dwarf-ext]: https://refspecs.linuxfoundation.org/LSB_5.0.0/LSB-Core-generic/LSB-Core-generic/dwarfext.html
unsafe fn read_encoded_pointer(
    reader: &mut DwarfReader,
    context: &EHContext<'_>,
    encoding: u8,
) -> Result<*const u8, ()> {
    if encoding == DW_EH_PE_omit {
        return Err(());
    }

    let base_ptr = match encoding & 0x70 {
        DW_EH_PE_absptr => core::ptr::null(),
        // relative to address of the encoded value, despite the name
        DW_EH_PE_pcrel => reader.ptr,
        DW_EH_PE_funcrel => {
            if context.func_start.is_null() {
                return Err(());
            }
            context.func_start
        }
        DW_EH_PE_textrel => (*context.get_text_start)(),
        DW_EH_PE_datarel => (*context.get_data_start)(),
        // aligned means the value is aligned to the size of a pointer
        DW_EH_PE_aligned => {
            reader.ptr = reader.ptr.with_addr(round_up(reader.ptr.addr(), size_of::<*const u8>())?);
            core::ptr::null()
        }
        _ => return Err(()),
    };

    let mut ptr = if base_ptr.is_null() {
        // any value encoding other than absptr would be nonsensical here;
        // there would be no source of pointer provenance
        if encoding & 0x0F != DW_EH_PE_absptr {
            return Err(());
        }
        unsafe { reader.read::<*const u8>() }
    } else {
        let offset = unsafe { read_encoded_offset(reader, encoding & 0x0F)? };
        base_ptr.wrapping_add(offset)
    };

    if encoding & DW_EH_PE_indirect != 0 {
        ptr = unsafe { *(ptr.cast::<*const u8>()) };
    }

    Ok(ptr)
}
