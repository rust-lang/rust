//@ compile-flags: -Zmir-opt-level=2 -Cpanic=abort
//@ only-64bit
//@ skip-filecheck

enum State {
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    S12,
    S13,
    S14,
    S15,
    S16,
    S17,
    S18,
    S19,
    S20,
    S21,
    S22,
    S23,
    S24,
    S25,
    S26,
    S27,
    S28,
    S29,
    S30,
    S31,
    S32,
    S33,
    S34,
    S35,
    S36,
    S37,
    S38,
    S39,
    S40,
    S41,
    S42,
    S43,
    S44,
    S45,
    S46,
    S47,
    S48,
    S49,
    S50,
    S51,
    S52,
    S53,
    S54,
    S55,
    S56,
    S57,
    S58,
    S59,
    S60,
    S61,
    S62,
    S63,
    S64,
    S65,
    S66,
    S67,
    S68,
    S69,
    S70,
    S71,
    S72,
    S73,
    S74,
    S75,
    S76,
    S77,
    S78,
    S79,
    S80,
    S81,
    S82,
    S83,
    S84,
    S85,
    S86,
    S87,
    S88,
    S89,
    S90,
    S91,
    S92,
    S93,
    S94,
    S95,
    S96,
    S97,
    S98,
    S99,
    S100,
    S101,
    S102,
    S103,
    S104,
    S105,
    S106,
    S107,
    S108,
    S109,
    S110,
    S111,
    S112,
    S113,
    S114,
    S115,
    S116,
    S117,
    S118,
    S119,
    S120,
    S121,
    S122,
    S123,
    S124,
    S125,
    S126,
    S127,
    S128,
    S129,
    S130,
    S131,
    S132,
}
use State::*;

pub fn autolink_email(s: &[u8]) -> Option<usize> {
    let mut cursor = 0;
    let mut marker = 0;
    let len = s.len();

    #[allow(unused_assignments)]
    let mut yych: u8 = 0;
    let mut yystate: State = S0;
    #[cfg_attr(feature = "loop_match", loop_match)]
    loop {
        yystate = 'blk: {
            match yystate {
                S0 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    cursor += 1;
                    match yych {
                        0x21
                        | 0x23..=0x27
                        | 0x2A..=0x2B
                        | 0x2D..=0x39
                        | 0x3D
                        | 0x3F
                        | 0x41..=0x5A
                        | 0x5E..=0x7E => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S3;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S1;
                        }
                    }
                }
                S1 => {
                    #[cfg_attr(feature = "loop_match", const_continue)]
                    break 'blk S2;
                }
                S2 => {
                    return Some(0); //return None;
                }
                S3 => {
                    marker = cursor;
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x21
                        | 0x23..=0x27
                        | 0x2A..=0x2B
                        | 0x2D..=0x39
                        | 0x3D
                        | 0x3F..=0x5A
                        | 0x5E..=0x7E => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S5;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S2;
                        }
                    }
                }
                S4 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    #[cfg_attr(feature = "loop_match", const_continue)]
                    break 'blk S5;
                }
                S5 => match yych {
                    0x21
                    | 0x23..=0x27
                    | 0x2A..=0x2B
                    | 0x2D..=0x39
                    | 0x3D
                    | 0x3F
                    | 0x41..=0x5A
                    | 0x5E..=0x7E => {
                        cursor += 1;
                        #[cfg_attr(feature = "loop_match", const_continue)]
                        break 'blk S4;
                    }
                    0x40 => {
                        cursor += 1;
                        #[cfg_attr(feature = "loop_match", const_continue)]
                        break 'blk S7;
                    }
                    _ => {
                        #[cfg_attr(feature = "loop_match", const_continue)]
                        break 'blk S6;
                    }
                },
                S6 => {
                    cursor = marker;
                    #[cfg_attr(feature = "loop_match", const_continue)]
                    break 'blk S2;
                }
                S7 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S8;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S8 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S9;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S10;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S9 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S12;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S13;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S10 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S12;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S13;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S11 => {
                    return Some(cursor);
                }
                S12 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S14;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S15;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S13 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S14;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S15;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S14 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S16;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S17;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S15 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S16;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S17;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S16 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S18;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S19;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S17 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S18;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S19;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S18 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S20;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S21;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S19 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S20;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S21;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S20 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S22;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S23;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S21 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S22;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S23;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S22 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S24;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S25;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S23 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S24;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S25;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S24 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S26;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S27;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S25 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S26;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S27;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S26 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S28;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S29;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S27 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S28;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S29;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S28 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S30;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S31;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S29 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S30;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S31;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S30 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S32;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S33;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S31 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S32;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S33;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S32 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S34;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S35;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S33 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S34;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S35;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S34 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S36;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S37;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S35 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S36;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S37;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S36 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S38;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S39;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S37 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S38;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S39;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S38 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S40;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S41;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S39 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S40;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S41;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S40 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S42;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S43;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S41 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S42;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S43;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S42 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S44;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S45;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S43 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S44;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S45;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S44 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S46;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S47;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S45 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S46;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S47;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S46 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S48;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S49;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S47 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S48;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S49;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S48 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S50;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S51;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S49 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S50;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S51;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S50 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S52;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S53;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S51 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S52;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S53;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S52 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S54;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S55;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S53 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S54;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S55;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S54 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S56;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S57;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S55 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S56;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S57;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S56 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S58;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S59;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S57 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S58;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S59;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S58 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S60;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S61;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S59 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S60;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S61;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S60 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S62;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S63;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S61 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S62;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S63;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S62 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S64;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S65;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S63 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S64;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S65;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S64 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S66;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S67;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S65 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S66;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S67;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S66 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S68;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S69;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S67 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S68;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S69;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S68 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S70;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S71;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S69 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S70;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S71;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S70 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S72;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S73;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S71 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S72;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S73;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S72 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S74;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S75;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S73 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S74;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S75;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S74 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S76;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S77;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S75 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S76;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S77;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S76 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S78;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S79;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S77 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S78;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S79;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S78 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S80;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S81;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S79 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S80;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S81;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S80 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S82;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S83;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S81 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S82;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S83;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S82 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S84;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S85;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S83 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S84;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S85;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S84 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S86;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S87;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S85 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S86;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S87;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S86 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S88;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S89;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S87 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S88;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S89;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S88 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S90;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S91;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S89 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S90;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S91;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S90 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S92;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S93;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S91 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S92;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S93;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S92 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S94;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S95;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S93 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S94;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S95;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S94 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S96;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S97;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S95 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S96;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S97;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S96 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S98;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S99;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S97 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S98;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S99;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S98 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S100;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S101;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S99 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S100;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S101;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S100 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S102;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S103;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S101 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S102;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S103;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S102 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S104;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S105;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S103 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S104;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S105;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S104 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S106;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S107;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S105 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S106;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S107;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S106 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S108;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S109;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S107 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S108;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S109;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S108 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S110;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S111;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S109 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S110;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S111;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S110 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S112;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S113;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S111 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S112;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S113;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S112 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S114;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S115;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S113 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S114;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S115;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S114 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S116;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S117;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S115 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S116;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S117;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S116 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S118;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S119;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S117 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S118;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S119;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S118 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S120;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S121;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S119 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S120;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S121;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S120 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S122;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S123;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S121 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S122;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S123;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S122 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S124;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S125;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S123 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S124;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S125;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S124 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S126;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S127;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S125 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S126;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S127;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S126 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S128;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S129;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S127 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S128;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S129;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S128 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S130;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S131;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S129 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2D => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S130;
                        }
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S131;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S130 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S132;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S131 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S132;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
                S132 => {
                    yych = unsafe { if cursor < len { *s.get_unchecked(cursor) } else { 0 } };
                    match yych {
                        0x2E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S7;
                        }
                        0x3E => {
                            cursor += 1;
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S11;
                        }
                        _ => {
                            #[cfg_attr(feature = "loop_match", const_continue)]
                            break 'blk S6;
                        }
                    }
                }
            }
        }
    }
}

// EMIT_MIR email_parser.autolink_email.JumpThreading.diff
// EMIT_MIR email_parser.autolink_email.runtime-optimized.after.mir
