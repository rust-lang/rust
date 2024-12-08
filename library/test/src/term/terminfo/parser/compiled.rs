#![allow(non_upper_case_globals, missing_docs)]

//! ncurses-compatible compiled terminfo format parsing (term(5))

use std::collections::HashMap;
use std::io;
use std::io::prelude::*;

use super::super::TermInfo;

#[cfg(test)]
mod tests;

// These are the orders ncurses uses in its compiled format (as of 5.9). Not sure if portable.

#[rustfmt::skip]
pub(crate) static boolfnames: &[&str] = &["auto_left_margin", "auto_right_margin",
    "no_esc_ctlc", "ceol_standout_glitch", "eat_newline_glitch", "erase_overstrike", "generic_type",
    "hard_copy", "has_meta_key", "has_status_line", "insert_null_glitch", "memory_above",
    "memory_below", "move_insert_mode", "move_standout_mode", "over_strike", "status_line_esc_ok",
    "dest_tabs_magic_smso", "tilde_glitch", "transparent_underline", "xon_xoff", "needs_xon_xoff",
    "prtr_silent", "hard_cursor", "non_rev_rmcup", "no_pad_char", "non_dest_scroll_region",
    "can_change", "back_color_erase", "hue_lightness_saturation", "col_addr_glitch",
    "cr_cancels_micro_mode", "has_print_wheel", "row_addr_glitch", "semi_auto_right_margin",
    "cpi_changes_res", "lpi_changes_res", "backspaces_with_bs", "crt_no_scrolling",
    "no_correctly_working_cr", "gnu_has_meta_key", "linefeed_is_newline", "has_hardware_tabs",
    "return_does_clr_eol"];

#[rustfmt::skip]
pub(crate) static boolnames: &[&str] = &["bw", "am", "xsb", "xhp", "xenl", "eo",
    "gn", "hc", "km", "hs", "in", "db", "da", "mir", "msgr", "os", "eslok", "xt", "hz", "ul", "xon",
    "nxon", "mc5i", "chts", "nrrmc", "npc", "ndscr", "ccc", "bce", "hls", "xhpa", "crxm", "daisy",
    "xvpa", "sam", "cpix", "lpix", "OTbs", "OTns", "OTnc", "OTMT", "OTNL", "OTpt", "OTxr"];

#[rustfmt::skip]
pub(crate) static numfnames: &[&str] = &[ "columns", "init_tabs", "lines",
    "lines_of_memory", "magic_cookie_glitch", "padding_baud_rate", "virtual_terminal",
    "width_status_line", "num_labels", "label_height", "label_width", "max_attributes",
    "maximum_windows", "max_colors", "max_pairs", "no_color_video", "buffer_capacity",
    "dot_vert_spacing", "dot_horz_spacing", "max_micro_address", "max_micro_jump", "micro_col_size",
    "micro_line_size", "number_of_pins", "output_res_char", "output_res_line",
    "output_res_horz_inch", "output_res_vert_inch", "print_rate", "wide_char_size", "buttons",
    "bit_image_entwining", "bit_image_type", "magic_cookie_glitch_ul", "carriage_return_delay",
    "new_line_delay", "backspace_delay", "horizontal_tab_delay", "number_of_function_keys"];

#[rustfmt::skip]
pub(crate) static numnames: &[&str] = &[ "cols", "it", "lines", "lm", "xmc", "pb",
    "vt", "wsl", "nlab", "lh", "lw", "ma", "wnum", "colors", "pairs", "ncv", "bufsz", "spinv",
    "spinh", "maddr", "mjump", "mcs", "mls", "npins", "orc", "orl", "orhi", "orvi", "cps", "widcs",
    "btns", "bitwin", "bitype", "UTug", "OTdC", "OTdN", "OTdB", "OTdT", "OTkn"];

#[rustfmt::skip]
pub(crate) static stringfnames: &[&str] = &[ "back_tab", "bell", "carriage_return",
    "change_scroll_region", "clear_all_tabs", "clear_screen", "clr_eol", "clr_eos",
    "column_address", "command_character", "cursor_address", "cursor_down", "cursor_home",
    "cursor_invisible", "cursor_left", "cursor_mem_address", "cursor_normal", "cursor_right",
    "cursor_to_ll", "cursor_up", "cursor_visible", "delete_character", "delete_line",
    "dis_status_line", "down_half_line", "enter_alt_charset_mode", "enter_blink_mode",
    "enter_bold_mode", "enter_ca_mode", "enter_delete_mode", "enter_dim_mode", "enter_insert_mode",
    "enter_secure_mode", "enter_protected_mode", "enter_reverse_mode", "enter_standout_mode",
    "enter_underline_mode", "erase_chars", "exit_alt_charset_mode", "exit_attribute_mode",
    "exit_ca_mode", "exit_delete_mode", "exit_insert_mode", "exit_standout_mode",
    "exit_underline_mode", "flash_screen", "form_feed", "from_status_line", "init_1string",
    "init_2string", "init_3string", "init_file", "insert_character", "insert_line",
    "insert_padding", "key_backspace", "key_catab", "key_clear", "key_ctab", "key_dc", "key_dl",
    "key_down", "key_eic", "key_eol", "key_eos", "key_f0", "key_f1", "key_f10", "key_f2", "key_f3",
    "key_f4", "key_f5", "key_f6", "key_f7", "key_f8", "key_f9", "key_home", "key_ic", "key_il",
    "key_left", "key_ll", "key_npage", "key_ppage", "key_right", "key_sf", "key_sr", "key_stab",
    "key_up", "keypad_local", "keypad_xmit", "lab_f0", "lab_f1", "lab_f10", "lab_f2", "lab_f3",
    "lab_f4", "lab_f5", "lab_f6", "lab_f7", "lab_f8", "lab_f9", "meta_off", "meta_on", "newline",
    "pad_char", "parm_dch", "parm_delete_line", "parm_down_cursor", "parm_ich", "parm_index",
    "parm_insert_line", "parm_left_cursor", "parm_right_cursor", "parm_rindex", "parm_up_cursor",
    "pkey_key", "pkey_local", "pkey_xmit", "print_screen", "prtr_off", "prtr_on", "repeat_char",
    "reset_1string", "reset_2string", "reset_3string", "reset_file", "restore_cursor",
    "row_address", "save_cursor", "scroll_forward", "scroll_reverse", "set_attributes", "set_tab",
    "set_window", "tab", "to_status_line", "underline_char", "up_half_line", "init_prog", "key_a1",
    "key_a3", "key_b2", "key_c1", "key_c3", "prtr_non", "char_padding", "acs_chars", "plab_norm",
    "key_btab", "enter_xon_mode", "exit_xon_mode", "enter_am_mode", "exit_am_mode", "xon_character",
    "xoff_character", "ena_acs", "label_on", "label_off", "key_beg", "key_cancel", "key_close",
    "key_command", "key_copy", "key_create", "key_end", "key_enter", "key_exit", "key_find",
    "key_help", "key_mark", "key_message", "key_move", "key_next", "key_open", "key_options",
    "key_previous", "key_print", "key_redo", "key_reference", "key_refresh", "key_replace",
    "key_restart", "key_resume", "key_save", "key_suspend", "key_undo", "key_sbeg", "key_scancel",
    "key_scommand", "key_scopy", "key_screate", "key_sdc", "key_sdl", "key_select", "key_send",
    "key_seol", "key_sexit", "key_sfind", "key_shelp", "key_shome", "key_sic", "key_sleft",
    "key_smessage", "key_smove", "key_snext", "key_soptions", "key_sprevious", "key_sprint",
    "key_sredo", "key_sreplace", "key_sright", "key_srsume", "key_ssave", "key_ssuspend",
    "key_sundo", "req_for_input", "key_f11", "key_f12", "key_f13", "key_f14", "key_f15", "key_f16",
    "key_f17", "key_f18", "key_f19", "key_f20", "key_f21", "key_f22", "key_f23", "key_f24",
    "key_f25", "key_f26", "key_f27", "key_f28", "key_f29", "key_f30", "key_f31", "key_f32",
    "key_f33", "key_f34", "key_f35", "key_f36", "key_f37", "key_f38", "key_f39", "key_f40",
    "key_f41", "key_f42", "key_f43", "key_f44", "key_f45", "key_f46", "key_f47", "key_f48",
    "key_f49", "key_f50", "key_f51", "key_f52", "key_f53", "key_f54", "key_f55", "key_f56",
    "key_f57", "key_f58", "key_f59", "key_f60", "key_f61", "key_f62", "key_f63", "clr_bol",
    "clear_margins", "set_left_margin", "set_right_margin", "label_format", "set_clock",
    "display_clock", "remove_clock", "create_window", "goto_window", "hangup", "dial_phone",
    "quick_dial", "tone", "pulse", "flash_hook", "fixed_pause", "wait_tone", "user0", "user1",
    "user2", "user3", "user4", "user5", "user6", "user7", "user8", "user9", "orig_pair",
    "orig_colors", "initialize_color", "initialize_pair", "set_color_pair", "set_foreground",
    "set_background", "change_char_pitch", "change_line_pitch", "change_res_horz",
    "change_res_vert", "define_char", "enter_doublewide_mode", "enter_draft_quality",
    "enter_italics_mode", "enter_leftward_mode", "enter_micro_mode", "enter_near_letter_quality",
    "enter_normal_quality", "enter_shadow_mode", "enter_subscript_mode", "enter_superscript_mode",
    "enter_upward_mode", "exit_doublewide_mode", "exit_italics_mode", "exit_leftward_mode",
    "exit_micro_mode", "exit_shadow_mode", "exit_subscript_mode", "exit_superscript_mode",
    "exit_upward_mode", "micro_column_address", "micro_down", "micro_left", "micro_right",
    "micro_row_address", "micro_up", "order_of_pins", "parm_down_micro", "parm_left_micro",
    "parm_right_micro", "parm_up_micro", "select_char_set", "set_bottom_margin",
    "set_bottom_margin_parm", "set_left_margin_parm", "set_right_margin_parm", "set_top_margin",
    "set_top_margin_parm", "start_bit_image", "start_char_set_def", "stop_bit_image",
    "stop_char_set_def", "subscript_characters", "superscript_characters", "these_cause_cr",
    "zero_motion", "char_set_names", "key_mouse", "mouse_info", "req_mouse_pos", "get_mouse",
    "set_a_foreground", "set_a_background", "pkey_plab", "device_type", "code_set_init",
    "set0_des_seq", "set1_des_seq", "set2_des_seq", "set3_des_seq", "set_lr_margin",
    "set_tb_margin", "bit_image_repeat", "bit_image_newline", "bit_image_carriage_return",
    "color_names", "define_bit_image_region", "end_bit_image_region", "set_color_band",
    "set_page_length", "display_pc_char", "enter_pc_charset_mode", "exit_pc_charset_mode",
    "enter_scancode_mode", "exit_scancode_mode", "pc_term_options", "scancode_escape",
    "alt_scancode_esc", "enter_horizontal_hl_mode", "enter_left_hl_mode", "enter_low_hl_mode",
    "enter_right_hl_mode", "enter_top_hl_mode", "enter_vertical_hl_mode", "set_a_attributes",
    "set_pglen_inch", "termcap_init2", "termcap_reset", "linefeed_if_not_lf", "backspace_if_not_bs",
    "other_non_function_keys", "arrow_key_map", "acs_ulcorner", "acs_llcorner", "acs_urcorner",
    "acs_lrcorner", "acs_ltee", "acs_rtee", "acs_btee", "acs_ttee", "acs_hline", "acs_vline",
    "acs_plus", "memory_lock", "memory_unlock", "box_chars_1"];

#[rustfmt::skip]
pub(crate) static stringnames: &[&str] = &[ "cbt", "_", "cr", "csr", "tbc", "clear",
    "_", "_", "hpa", "cmdch", "cup", "cud1", "home", "civis", "cub1", "mrcup", "cnorm", "cuf1",
    "ll", "cuu1", "cvvis", "dch1", "dl1", "dsl", "hd", "smacs", "blink", "bold", "smcup", "smdc",
    "dim", "smir", "invis", "prot", "rev", "smso", "smul", "ech", "rmacs", "sgr0", "rmcup", "rmdc",
    "rmir", "rmso", "rmul", "flash", "ff", "fsl", "is1", "is2", "is3", "if", "ich1", "il1", "ip",
    "kbs", "ktbc", "kclr", "kctab", "_", "_", "kcud1", "_", "_", "_", "_", "_", "_", "_", "_", "_",
    "_", "_", "_", "_", "_", "khome", "_", "_", "kcub1", "_", "knp", "kpp", "kcuf1", "_", "_",
    "khts", "_", "rmkx", "smkx", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "rmm", "_",
    "_", "pad", "dch", "dl", "cud", "ich", "indn", "il", "cub", "cuf", "rin", "cuu", "pfkey",
    "pfloc", "pfx", "mc0", "mc4", "_", "rep", "rs1", "rs2", "rs3", "rf", "rc", "vpa", "sc", "ind",
    "ri", "sgr", "_", "wind", "_", "tsl", "uc", "hu", "iprog", "_", "_", "_", "_", "_", "mc5p",
    "rmp", "acsc", "pln", "kcbt", "smxon", "rmxon", "smam", "rmam", "xonc", "xoffc", "_", "smln",
    "rmln", "_", "kcan", "kclo", "kcmd", "kcpy", "kcrt", "_", "kent", "kext", "kfnd", "khlp",
    "kmrk", "kmsg", "kmov", "knxt", "kopn", "kopt", "kprv", "kprt", "krdo", "kref", "krfr", "krpl",
    "krst", "kres", "ksav", "kspd", "kund", "kBEG", "kCAN", "kCMD", "kCPY", "kCRT", "_", "_",
    "kslt", "kEND", "kEOL", "kEXT", "kFND", "kHLP", "kHOM", "_", "kLFT", "kMSG", "kMOV", "kNXT",
    "kOPT", "kPRV", "kPRT", "kRDO", "kRPL", "kRIT", "kRES", "kSAV", "kSPD", "kUND", "rfi", "_", "_",
    "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_",
    "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_",
    "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_",
    "dclk", "rmclk", "cwin", "wingo", "_", "dial", "qdial", "_", "_", "hook", "pause", "wait", "_",
    "_", "_", "_", "_", "_", "_", "_", "_", "_", "op", "oc", "initc", "initp", "scp", "setf",
    "setb", "cpi", "lpi", "chr", "cvr", "defc", "swidm", "sdrfq", "sitm", "slm", "smicm", "snlq",
    "snrmq", "sshm", "ssubm", "ssupm", "sum", "rwidm", "ritm", "rlm", "rmicm", "rshm", "rsubm",
    "rsupm", "rum", "mhpa", "mcud1", "mcub1", "mcuf1", "mvpa", "mcuu1", "porder", "mcud", "mcub",
    "mcuf", "mcuu", "scs", "smgb", "smgbp", "smglp", "smgrp", "smgt", "smgtp", "sbim", "scsd",
    "rbim", "rcsd", "subcs", "supcs", "docr", "zerom", "csnm", "kmous", "minfo", "reqmp", "getm",
    "setaf", "setab", "pfxl", "devt", "csin", "s0ds", "s1ds", "s2ds", "s3ds", "smglr", "smgtb",
    "birep", "binel", "bicr", "colornm", "defbi", "endbi", "setcolor", "slines", "dispc", "smpch",
    "rmpch", "smsc", "rmsc", "pctrm", "scesc", "scesa", "ehhlm", "elhlm", "elohlm", "erhlm",
    "ethlm", "evhlm", "sgr1", "slength", "OTi2", "OTrs", "OTnl", "OTbs", "OTko", "OTma", "OTG2",
    "OTG3", "OTG1", "OTG4", "OTGR", "OTGL", "OTGU", "OTGD", "OTGH", "OTGV", "OTGC", "meml", "memu",
    "box1"];

fn read_le_u16(r: &mut dyn io::Read) -> io::Result<u16> {
    let mut b = [0; 2];
    r.read_exact(&mut b)?;
    Ok((b[0] as u16) | ((b[1] as u16) << 8))
}

fn read_le_u32(r: &mut dyn io::Read) -> io::Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok((b[0] as u32) | ((b[1] as u32) << 8) | ((b[2] as u32) << 16) | ((b[3] as u32) << 24))
}

fn read_byte(r: &mut dyn io::Read) -> io::Result<u8> {
    match r.bytes().next() {
        Some(s) => s,
        None => Err(io::Error::new(io::ErrorKind::Other, "end of file")),
    }
}

/// Parse a compiled terminfo entry, using long capability names if `longnames`
/// is true
pub(crate) fn parse(file: &mut dyn io::Read, longnames: bool) -> Result<TermInfo, String> {
    macro_rules! t( ($e:expr) => (
        match $e {
            Ok(e) => e,
            Err(e) => return Err(e.to_string())
        }
    ) );

    let (bnames, snames, nnames) = if longnames {
        (boolfnames, stringfnames, numfnames)
    } else {
        (boolnames, stringnames, numnames)
    };

    // Check magic number
    let magic = t!(read_le_u16(file));

    let extended = match magic {
        0o0432 => false,
        0o01036 => true,
        _ => return Err(format!("invalid magic number, found {magic:o}")),
    };

    // According to the spec, these fields must be >= -1 where -1 means that the feature is not
    // supported. Using 0 instead of -1 works because we skip sections with length 0.
    macro_rules! read_nonneg {
        () => {{
            match t!(read_le_u16(file)) as i16 {
                n if n >= 0 => n as usize,
                -1 => 0,
                _ => return Err("incompatible file: length fields must be  >= -1".to_string()),
            }
        }};
    }

    let names_bytes = read_nonneg!();
    let bools_bytes = read_nonneg!();
    let numbers_count = read_nonneg!();
    let string_offsets_count = read_nonneg!();
    let string_table_bytes = read_nonneg!();

    if names_bytes == 0 {
        return Err("incompatible file: names field must be at least 1 byte wide".to_string());
    }

    if bools_bytes > boolnames.len() {
        return Err("incompatible file: more booleans than expected".to_string());
    }

    if numbers_count > numnames.len() {
        return Err("incompatible file: more numbers than expected".to_string());
    }

    if string_offsets_count > stringnames.len() {
        return Err("incompatible file: more string offsets than expected".to_string());
    }

    // don't read NUL
    let mut bytes = Vec::new();
    t!(file.take((names_bytes - 1) as u64).read_to_end(&mut bytes));
    let names_str = match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return Err("input not utf-8".to_string()),
    };

    let term_names: Vec<String> = names_str.split('|').map(|s| s.to_string()).collect();
    // consume NUL
    if t!(read_byte(file)) != b'\0' {
        return Err("incompatible file: missing null terminator for names section".to_string());
    }

    let bools_map: HashMap<String, bool> = t! {
        (0..bools_bytes).filter_map(|i| match read_byte(file) {
            Err(e) => Some(Err(e)),
            Ok(1) => Some(Ok((bnames[i].to_string(), true))),
            Ok(_) => None
        }).collect()
    };

    if (bools_bytes + names_bytes) % 2 == 1 {
        t!(read_byte(file)); // compensate for padding
    }

    let numbers_map: HashMap<String, u32> = t! {
        (0..numbers_count).filter_map(|i| {
            let number = if extended { read_le_u32(file) } else { read_le_u16(file).map(Into::into) };

            match number {
                Ok(0xFFFF) => None,
                Ok(n) => Some(Ok((nnames[i].to_string(), n))),
                Err(e) => Some(Err(e))
            }
        }).collect()
    };

    let string_map: HashMap<String, Vec<u8>> = if string_offsets_count > 0 {
        let string_offsets: Vec<u16> =
            t!((0..string_offsets_count).map(|_| read_le_u16(file)).collect());

        let mut string_table = Vec::new();
        t!(file.take(string_table_bytes as u64).read_to_end(&mut string_table));

        t!(string_offsets
            .into_iter()
            .enumerate()
            .filter(|&(_, offset)| {
                // non-entry
                offset != 0xFFFF
            })
            .map(|(i, offset)| {
                let offset = offset as usize;

                let name = if snames[i] == "_" { stringfnames[i] } else { snames[i] };

                if offset == 0xFFFE {
                    // undocumented: FFFE indicates cap@, which means the capability is not present
                    // unsure if the handling for this is correct
                    return Ok((name.to_string(), Vec::new()));
                }

                // Find the offset of the NUL we want to go to
                let nulpos = string_table[offset..string_table_bytes].iter().position(|&b| b == 0);
                match nulpos {
                    Some(len) => {
                        Ok((name.to_string(), string_table[offset..offset + len].to_vec()))
                    }
                    None => Err("invalid file: missing NUL in string_table".to_string()),
                }
            })
            .collect())
    } else {
        HashMap::new()
    };

    // And that's all there is to it
    Ok(TermInfo { names: term_names, bools: bools_map, numbers: numbers_map, strings: string_map })
}

/// Creates a dummy TermInfo struct for msys terminals
pub(crate) fn msys_terminfo() -> TermInfo {
    let mut strings = HashMap::new();
    strings.insert("sgr0".to_string(), b"\x1B[0m".to_vec());
    strings.insert("bold".to_string(), b"\x1B[1m".to_vec());
    strings.insert("setaf".to_string(), b"\x1B[3%p1%dm".to_vec());
    strings.insert("setab".to_string(), b"\x1B[4%p1%dm".to_vec());

    let mut numbers = HashMap::new();
    numbers.insert("colors".to_string(), 8);

    TermInfo {
        names: vec!["cygwin".to_string()], // msys is a fork of an older cygwin version
        bools: HashMap::new(),
        numbers,
        strings,
    }
}
