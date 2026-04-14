use super::*;
use core::mem::{align_of, size_of};

// Key::from_raw tests
#[test]
fn key_from_raw_letters() {
    assert_eq!(Key::from_raw(0x04), Key::A);
    assert_eq!(Key::from_raw(0x1D), Key::Z);
    assert_eq!(Key::from_raw(0x10), Key::M);
}

#[test]
fn key_from_raw_numbers() {
    assert_eq!(Key::from_raw(0x1E), Key::Num1);
    assert_eq!(Key::from_raw(0x27), Key::Num0);
}

#[test]
fn key_from_raw_special() {
    assert_eq!(Key::from_raw(0x28), Key::Enter);
    assert_eq!(Key::from_raw(0x29), Key::Escape);
    assert_eq!(Key::from_raw(0x2A), Key::Backspace);
    assert_eq!(Key::from_raw(0x2C), Key::Space);
}

#[test]
fn key_from_raw_function_keys() {
    assert_eq!(Key::from_raw(0x3A), Key::F1);
    assert_eq!(Key::from_raw(0x45), Key::F12);
}

#[test]
fn key_from_raw_modifiers() {
    assert_eq!(Key::from_raw(0xE0), Key::LeftCtrl);
    assert_eq!(Key::from_raw(0xE1), Key::LeftShift);
    assert_eq!(Key::from_raw(0xE4), Key::RightCtrl);
    assert_eq!(Key::from_raw(0xE7), Key::RightMeta);
}

#[test]
fn key_from_raw_unknown_fallback() {
    assert_eq!(Key::from_raw(0xFF), Key::Unknown);
    assert_eq!(Key::from_raw(0x00), Key::Unknown);
    assert_eq!(Key::from_raw(0xFFFE), Key::Unknown);
}

// Key::name tests
#[test]
fn key_name_returns_correct_string() {
    assert_eq!(Key::A.name(), "A");
    assert_eq!(Key::Enter.name(), "Enter");
    assert_eq!(Key::Space.name(), "Space");
    assert_eq!(Key::F1.name(), "F1");
    assert_eq!(Key::LeftCtrl.name(), "LCtrl");
    assert_eq!(Key::Unknown.name(), "?");
}

// Mods tests
#[test]
fn mods_has_shift() {
    let mods = Mods(Mods::SHIFT);
    assert!(mods.has_shift());
    assert!(!mods.has_ctrl());
    assert!(!mods.has_alt());
    assert!(!mods.has_meta());
}

#[test]
fn mods_combined() {
    let mods = Mods(Mods::SHIFT | Mods::CTRL | Mods::ALT);
    assert!(mods.has_shift());
    assert!(mods.has_ctrl());
    assert!(mods.has_alt());
    assert!(!mods.has_meta());
}

#[test]
fn mods_empty() {
    let mods = Mods(0);
    assert!(!mods.has_shift());
    assert!(!mods.has_ctrl());
    assert!(!mods.has_alt());
    assert!(!mods.has_meta());
}

#[test]
fn mods_bitflag_algebra() {
    let mods = Mods(Mods::SHIFT | Mods::CTRL | Mods::ALT);
    assert_eq!(Mods(mods.0 | mods.0), mods);
    assert_eq!(Mods(mods.0 & 0), Mods(0));

    let toggled = Mods(mods.0 ^ Mods::SHIFT);
    assert!(!toggled.has_shift());
    assert!(toggled.has_ctrl());
    assert!(toggled.has_alt());
}

#[test]
fn mods_roundtrip_byte() {
    let mods = Mods(Mods::SHIFT | Mods::META);
    let round = Mods(mods.0);
    assert_eq!(mods, round);
}

// KeyEventPayload tests
#[test]
fn key_event_payload_is_repeat() {
    let payload = KeyEventPayload {
        key: 0x04,
        mods: 0,
        flags: 0,
    };
    assert!(!payload.is_repeat());

    let payload_repeat = KeyEventPayload {
        key: 0x04,
        mods: 0,
        flags: 1,
    };
    assert!(payload_repeat.is_repeat());
}

#[test]
fn key_event_payload_key() {
    let payload = KeyEventPayload {
        key: 0x04,
        mods: 0,
        flags: 0,
    };
    assert_eq!(payload.key(), Key::A);
}

#[test]
fn key_event_payload_mods() {
    let payload = KeyEventPayload {
        key: 0x04,
        mods: Mods::SHIFT | Mods::CTRL,
        flags: 0,
    };
    let mods = payload.mods();
    assert!(mods.has_shift());
    assert!(mods.has_ctrl());
}

// Struct size checks for wire compatibility
#[test]
fn bristle_event_header_size() {
    assert_eq!(core::mem::size_of::<BristleEventHeader>(), 20);
}

#[test]
fn key_event_payload_size() {
    assert_eq!(core::mem::size_of::<KeyEventPayload>(), 4);
}

#[test]
fn raw_input_envelope_size() {
    assert_eq!(core::mem::size_of::<RawInputEnvelope>(), 24);
}

#[test]
fn ps2_key_payload_size() {
    assert_eq!(core::mem::size_of::<Ps2KeyPayload>(), 2);
}

#[test]
fn pointer_move_payload_size() {
    assert_eq!(core::mem::size_of::<PointerMovePayload>(), 4);
}

#[test]
fn pointer_button_payload_size() {
    assert_eq!(core::mem::size_of::<PointerButtonPayload>(), 2);
}

#[test]
fn scroll_payload_size() {
    assert_eq!(core::mem::size_of::<ScrollPayload>(), 4);
}

// Alignment checks for ABI stability
#[test]
fn bristle_event_header_alignment() {
    assert_eq!(align_of::<BristleEventHeader>(), 1);
}

#[test]
fn key_event_payload_alignment() {
    assert_eq!(align_of::<KeyEventPayload>(), 1);
}

#[test]
fn raw_input_envelope_alignment() {
    assert_eq!(align_of::<RawInputEnvelope>(), 1);
}

#[test]
fn pointer_payload_alignment() {
    assert_eq!(align_of::<PointerMovePayload>(), 1);
    assert_eq!(align_of::<ScrollPayload>(), 1);
}

// Magic and version constants
#[test]
fn bristle_event_magic_is_hide() {
    // 'HIDE' in ASCII = 0x48494445
    assert_eq!(BRISTLE_EVENT_MAGIC, 0x48494445);
}

#[test]
fn bristle_event_version_is_zero() {
    assert_eq!(BRISTLE_EVENT_VERSION, 0);
}

#[test]
fn bristle_event_header_golden_bytes() {
    let header = BristleEventHeader {
        magic: BRISTLE_EVENT_MAGIC,
        version: BRISTLE_EVENT_VERSION,
        event_type: EventType::KeyDown as u16,
        timestamp_ns: 0x1122_3344_5566_7788,
        payload_len: 4,
    };
    let bytes = header.to_bytes();
    let expected = [
        0x45, 0x44, 0x49, 0x48, // magic "HIDE" LE
        0x00, 0x00, // version
        0x01, 0x00, // event_type
        0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // timestamp
        0x04, 0x00, 0x00, 0x00, // payload_len
    ];
    assert_eq!(bytes, expected);
    let parsed = BristleEventHeader::from_bytes(&bytes).unwrap();
    let event_type = parsed.event_type;
    let timestamp_ns = parsed.timestamp_ns;
    let payload_len = parsed.payload_len;
    assert_eq!(event_type, EventType::KeyDown as u16);
    assert_eq!(timestamp_ns, 0x1122_3344_5566_7788);
    assert_eq!(payload_len, 4);
}

#[test]
fn bristle_event_header_rejects_bad_magic_and_version() {
    let mut bytes = BristleEventHeader {
        magic: BRISTLE_EVENT_MAGIC,
        version: BRISTLE_EVENT_VERSION,
        event_type: EventType::KeyUp as u16,
        timestamp_ns: 0,
        payload_len: 0,
    }
    .to_bytes();

    bytes[0] ^= 0xFF;
    assert!(matches!(
        BristleEventHeader::from_bytes(&bytes),
        Err(HidParseError::BadMagic)
    ));

    let mut bytes = BristleEventHeader {
        magic: BRISTLE_EVENT_MAGIC,
        version: BRISTLE_EVENT_VERSION,
        event_type: EventType::KeyUp as u16,
        timestamp_ns: 0,
        payload_len: 0,
    }
    .to_bytes();
    bytes[4] = 1;
    assert!(matches!(
        BristleEventHeader::from_bytes(&bytes),
        Err(HidParseError::BadVersion)
    ));
}

#[test]
fn bristle_event_header_versioning_policy() {
    let bytes = BristleEventHeader {
        magic: BRISTLE_EVENT_MAGIC,
        version: 1,
        event_type: EventType::KeyDown as u16,
        timestamp_ns: 0,
        payload_len: 0,
    }
    .to_bytes();
    assert!(matches!(
        BristleEventHeader::from_bytes(&bytes),
        Err(HidParseError::BadVersion)
    ));
}

#[test]
fn payload_golden_bytes() {
    let key_payload = KeyEventPayload {
        key: 0x04,
        mods: Mods::SHIFT | Mods::CTRL,
        flags: 1,
    };
    assert_eq!(key_payload.to_bytes(), [0x04, 0x00, 0x03, 0x01]);
    let parsed = KeyEventPayload::from_bytes(&key_payload.to_bytes());
    assert_eq!(parsed.key(), Key::A);
    assert!(parsed.mods().has_shift());
    assert!(parsed.mods().has_ctrl());
    assert!(parsed.is_repeat());

    let move_payload = PointerMovePayload { dx: -2, dy: 300 };
    assert_eq!(move_payload.to_bytes(), [0xfe, 0xff, 0x2c, 0x01]);
    let parsed = PointerMovePayload::from_bytes(&move_payload.to_bytes());
    let dx = parsed.dx;
    let dy = parsed.dy;
    assert_eq!(dx, -2);
    assert_eq!(dy, 300);

    let scroll_payload = ScrollPayload { dx: 120, dy: -120 };
    assert_eq!(scroll_payload.to_bytes(), [0x78, 0x00, 0x88, 0xff]);
    let parsed = ScrollPayload::from_bytes(&scroll_payload.to_bytes());
    let dx = parsed.dx;
    let dy = parsed.dy;
    assert_eq!(dx, 120);
    assert_eq!(dy, -120);
}

#[test]
fn raw_input_envelope_invariants() {
    let envelope = RawInputEnvelope {
        device_id: 0x1111_2222_3333_4444,
        timestamp_ns: 0x0102_0304_0506_0708,
        kind: InputDeviceKind::Keyboard as u8,
        payload_len: 2,
        _pad: [0u8; 6],
    };
    const TOTAL_LEN: usize = RawInputEnvelope::SIZE + 2;
    let mut bytes = [0u8; TOTAL_LEN];
    bytes[..RawInputEnvelope::SIZE].copy_from_slice(&envelope.to_bytes());
    bytes[RawInputEnvelope::SIZE..].copy_from_slice(&[0xaa, 0xbb]);
    let (parsed, payload) = RawInputEnvelope::from_bytes(&bytes).unwrap();
    let device_id = parsed.device_id;
    let payload_len = parsed.payload_len;
    let expected_device_id = envelope.device_id;
    assert_eq!(device_id, expected_device_id);
    assert_eq!(payload_len, 2);
    assert_eq!(payload, &[0xaa, 0xbb]);

    let bad = &bytes[..TOTAL_LEN - 1];
    assert!(matches!(
        RawInputEnvelope::from_bytes(&bad),
        Err(HidParseError::LengthMismatch)
    ));

    let mut bad_kind = bytes;
    bad_kind[16] = 0xff;
    assert!(matches!(
        RawInputEnvelope::from_bytes(&bad_kind),
        Err(HidParseError::BadKind)
    ));
}

#[test]
fn raw_input_envelope_length_bounds() {
    let mut bytes = [0u8; RawInputEnvelope::SIZE];
    bytes[16] = InputDeviceKind::Keyboard as u8;
    bytes[17] = 0;
    assert!(RawInputEnvelope::from_bytes(&bytes).is_ok());

    bytes[17] = 1;
    assert!(matches!(
        RawInputEnvelope::from_bytes(&bytes),
        Err(HidParseError::LengthMismatch)
    ));

    let mut max_payload = [0u8; RawInputEnvelope::SIZE + 255];
    max_payload[16] = InputDeviceKind::Keyboard as u8;
    max_payload[17] = 255;
    assert!(RawInputEnvelope::from_bytes(&max_payload).is_ok());

    let mut too_long = [0u8; RawInputEnvelope::SIZE + 256];
    too_long[..RawInputEnvelope::SIZE + 255].copy_from_slice(&max_payload);
    assert!(matches!(
        RawInputEnvelope::from_bytes(&too_long),
        Err(HidParseError::LengthMismatch)
    ));
}

#[test]
fn key_mapping_is_unique_and_named() {
    let known: &[(u16, Key)] = &[
        (0x04, Key::A),
        (0x05, Key::B),
        (0x06, Key::C),
        (0x07, Key::D),
        (0x08, Key::E),
        (0x09, Key::F),
        (0x0A, Key::G),
        (0x0B, Key::H),
        (0x0C, Key::I),
        (0x0D, Key::J),
        (0x0E, Key::K),
        (0x0F, Key::L),
        (0x10, Key::M),
        (0x11, Key::N),
        (0x12, Key::O),
        (0x13, Key::P),
        (0x14, Key::Q),
        (0x15, Key::R),
        (0x16, Key::S),
        (0x17, Key::T),
        (0x18, Key::U),
        (0x19, Key::V),
        (0x1A, Key::W),
        (0x1B, Key::X),
        (0x1C, Key::Y),
        (0x1D, Key::Z),
        (0x1E, Key::Num1),
        (0x1F, Key::Num2),
        (0x20, Key::Num3),
        (0x21, Key::Num4),
        (0x22, Key::Num5),
        (0x23, Key::Num6),
        (0x24, Key::Num7),
        (0x25, Key::Num8),
        (0x26, Key::Num9),
        (0x27, Key::Num0),
        (0x28, Key::Enter),
        (0x29, Key::Escape),
        (0x2A, Key::Backspace),
        (0x2B, Key::Tab),
        (0x2C, Key::Space),
        (0x2D, Key::Minus),
        (0x2E, Key::Equal),
        (0x2F, Key::LeftBracket),
        (0x30, Key::RightBracket),
        (0x31, Key::Backslash),
        (0x33, Key::Semicolon),
        (0x34, Key::Quote),
        (0x35, Key::Grave),
        (0x36, Key::Comma),
        (0x37, Key::Period),
        (0x38, Key::Slash),
        (0x39, Key::CapsLock),
        (0x3A, Key::F1),
        (0x3B, Key::F2),
        (0x3C, Key::F3),
        (0x3D, Key::F4),
        (0x3E, Key::F5),
        (0x3F, Key::F6),
        (0x40, Key::F7),
        (0x41, Key::F8),
        (0x42, Key::F9),
        (0x43, Key::F10),
        (0x44, Key::F11),
        (0x45, Key::F12),
        (0x46, Key::PrintScreen),
        (0x47, Key::ScrollLock),
        (0x48, Key::Pause),
        (0x49, Key::Insert),
        (0x4A, Key::Home),
        (0x4B, Key::PageUp),
        (0x4C, Key::Delete),
        (0x4D, Key::End),
        (0x4E, Key::PageDown),
        (0x4F, Key::Right),
        (0x50, Key::Left),
        (0x51, Key::Down),
        (0x52, Key::Up),
        (0xE0, Key::LeftCtrl),
        (0xE1, Key::LeftShift),
        (0xE2, Key::LeftAlt),
        (0xE3, Key::LeftMeta),
        (0xE4, Key::RightCtrl),
        (0xE5, Key::RightShift),
        (0xE6, Key::RightAlt),
        (0xE7, Key::RightMeta),
    ];

    for i in 0..known.len() {
        let (raw_i, key_i) = known[i];
        assert_eq!(Key::from_raw(raw_i), key_i);
        assert!(!key_i.name().is_empty());
        assert_ne!(key_i, Key::Unknown);
        for j in (i + 1)..known.len() {
            let (raw_j, key_j) = known[j];
            assert_ne!(raw_i, raw_j);
            assert_ne!(key_i, key_j);
        }
    }
}

#[test]
fn fuzz_lite_parsing_does_not_panic() {
    let mut seed: u64 = 0x1234_5678_9abc_def0;
    for _ in 0..1024 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut header_bytes = [0u8; BristleEventHeader::SIZE];
        for (idx, slot) in header_bytes.iter_mut().enumerate() {
            *slot = ((seed >> ((idx % 8) * 8)) & 0xff) as u8;
        }
        let _ = BristleEventHeader::from_bytes(&header_bytes);

        let len = (seed as usize) % 64;
        let mut envelope_bytes = [0u8; 64];
        for (idx, slot) in envelope_bytes.iter_mut().enumerate().take(len) {
            *slot = ((seed >> ((idx % 8) * 8)) & 0xff) as u8;
        }
        let parsed = RawInputEnvelope::from_bytes(&envelope_bytes[..len]);
        if let Ok((env, payload)) = parsed {
            assert_eq!(len, RawInputEnvelope::SIZE + env.payload_len as usize);
            assert_eq!(payload.len(), env.payload_len as usize);
            assert!(InputDeviceKind::from_raw(env.kind).is_ok());
        }
    }
}

#[test]
fn abi_layout_hash_guard() {
    macro_rules! offset_of {
        ($ty:ty, $field:ident) => {{
            let uninit = core::mem::MaybeUninit::<$ty>::uninit();
            let base = uninit.as_ptr();
            unsafe { (core::ptr::addr_of!((*base).$field) as usize) - (base as usize) }
        }};
    }

    fn fnv64(mut hash: u64, value: u64) -> u64 {
        const FNV_PRIME: u64 = 1099511628211;
        hash ^= value;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash
    }

    let mut hash = 0xcbf29ce484222325u64;

    hash = fnv64(hash, size_of::<BristleEventHeader>() as u64);
    hash = fnv64(hash, align_of::<BristleEventHeader>() as u64);
    hash = fnv64(hash, offset_of!(BristleEventHeader, magic) as u64);
    hash = fnv64(hash, offset_of!(BristleEventHeader, version) as u64);
    hash = fnv64(hash, offset_of!(BristleEventHeader, event_type) as u64);
    hash = fnv64(hash, offset_of!(BristleEventHeader, timestamp_ns) as u64);
    hash = fnv64(hash, offset_of!(BristleEventHeader, payload_len) as u64);

    hash = fnv64(hash, size_of::<KeyEventPayload>() as u64);
    hash = fnv64(hash, align_of::<KeyEventPayload>() as u64);
    hash = fnv64(hash, offset_of!(KeyEventPayload, key) as u64);
    hash = fnv64(hash, offset_of!(KeyEventPayload, mods) as u64);
    hash = fnv64(hash, offset_of!(KeyEventPayload, flags) as u64);

    hash = fnv64(hash, size_of::<RawInputEnvelope>() as u64);
    hash = fnv64(hash, align_of::<RawInputEnvelope>() as u64);
    hash = fnv64(hash, offset_of!(RawInputEnvelope, device_id) as u64);
    hash = fnv64(hash, offset_of!(RawInputEnvelope, timestamp_ns) as u64);
    hash = fnv64(hash, offset_of!(RawInputEnvelope, kind) as u64);
    hash = fnv64(hash, offset_of!(RawInputEnvelope, payload_len) as u64);

    hash = fnv64(hash, size_of::<PointerMovePayload>() as u64);
    hash = fnv64(hash, align_of::<PointerMovePayload>() as u64);
    hash = fnv64(hash, offset_of!(PointerMovePayload, dx) as u64);
    hash = fnv64(hash, offset_of!(PointerMovePayload, dy) as u64);

    hash = fnv64(hash, size_of::<ScrollPayload>() as u64);
    hash = fnv64(hash, align_of::<ScrollPayload>() as u64);
    hash = fnv64(hash, offset_of!(ScrollPayload, dx) as u64);
    hash = fnv64(hash, offset_of!(ScrollPayload, dy) as u64);

    assert_eq!(hash, 0x5791d02bd6fe2b5e);
}
