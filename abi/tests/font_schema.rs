//! Schema unit tests for font graph kinds, keys, and predicates.

#[test]
fn font_graph_kinds_exist() {
    use abi::schema::kinds;

    // Core font hierarchy
    assert!(!kinds::FONT_SUPERFAMILY.is_empty());
    assert!(!kinds::FONT_FAMILY.is_empty());
    assert!(!kinds::FONT_FACE.is_empty());
    assert!(!kinds::FONT_BLOB.is_empty());
    assert!(!kinds::FONT_FILE.is_empty());
    assert!(!kinds::FONT_ATLAS.is_empty());
    assert!(!kinds::FONT_COVERAGE.is_empty());
    assert!(!kinds::FONT_GLYPH.is_empty());

    // Request kinds
    assert!(!kinds::FONT_IMPORT_REQUEST.is_empty());
    assert!(!kinds::FONT_GLYPH_REQUEST.is_empty());
}

#[test]
fn font_graph_rels_exist() {
    use abi::schema::rels;

    // Hierarchy relationships
    assert!(!rels::FONT_CONTAINS.is_empty());
    assert!(!rels::FONT_HAS_FACE.is_empty());
    assert!(!rels::FONT_HAS_BLOB.is_empty());
    assert!(!rels::FONT_HAS_ASSET.is_empty());
    assert!(!rels::FONT_HAS_ATLAS.is_empty());
    assert!(!rels::FONT_HAS_COVERAGE.is_empty());
    assert!(!rels::FONT_HAS_GLYPH.is_empty());
    assert!(!rels::FONT_FALLBACK_TO.is_empty());
}

#[test]
fn font_graph_keys_exist() {
    use abi::schema::keys;

    // Font identity keys
    assert!(!keys::FONT_NAME.is_empty());
    assert!(!keys::FONT_FAMILY_KEY.is_empty());
    assert!(!keys::FONT_STYLE.is_empty());
    assert!(!keys::FONT_WEIGHT.is_empty());
    assert!(!keys::FONT_WIDTH.is_empty());
    assert!(!keys::FONT_SLOPE.is_empty());

    // Blob keys
    assert!(!keys::FONT_BYTESPACE.is_empty());
    assert!(!keys::FONT_BLOB_SHA256.is_empty());
    assert!(!keys::FONT_BLOB_MIME.is_empty());

    // Atlas keys
    assert!(!keys::FONT_ATLAS_BYTESPACE.is_empty());
    assert!(!keys::FONT_ATLAS_WIDTH.is_empty());
    assert!(!keys::FONT_ATLAS_HEIGHT.is_empty());
    assert!(!keys::FONT_ATLAS_FORMAT.is_empty());
    assert!(!keys::FONT_ATLAS_VERSION.is_empty());

    // Glyph keys
    assert!(!keys::FONT_GLYPH_CODEPOINT.is_empty());
    assert!(!keys::FONT_GLYPH_PX_SIZE.is_empty());
    assert!(!keys::FONT_GLYPH_BITMAP.is_empty());
    assert!(!keys::FONT_GLYPH_ADVANCE.is_empty());
}

#[test]
fn font_kinds_no_collision() {
    use abi::schema::kinds;
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    let kinds = [
        kinds::FONT_SUPERFAMILY,
        kinds::FONT_FAMILY,
        kinds::FONT_FACE,
        kinds::FONT_BLOB,
        kinds::FONT_FILE,
        kinds::FONT_ATLAS,
        kinds::FONT_COVERAGE,
        kinds::FONT_GLYPH,
        kinds::FONT_IMPORT_REQUEST,
        kinds::FONT_GLYPH_REQUEST,
    ];

    for k in kinds {
        assert!(seen.insert(k), "Duplicate kind: {}", k);
    }
}

#[test]
fn font_protocol_encode_decode_roundtrip() {
    use abi::font_protocol::*;
    use abi::ids::HandleId;
    use abi::wire::ThingId;

    // Test GetFaceMetrics
    let req = GetFaceMetrics {
        face_id: ThingId::from_u64(0x12345678),
        px_size: 16,
    };
    let mut buf = [0u8; 64];
    let len = req.encode(&mut buf).unwrap();
    let decoded = GetFaceMetrics::decode(&buf[1..len]).unwrap();
    assert_eq!(decoded.face_id.to_u64_lossy(), 0x12345678);
    assert_eq!(decoded.px_size, 16);

    // Test FaceMetrics
    let metrics = FaceMetrics {
        ascent: 800,
        descent: -200,
        line_gap: 100,
        units_per_em: 1000,
    };
    let mut buf = [0u8; 64];
    let len = metrics.encode(&mut buf).unwrap();
    let decoded = FaceMetrics::decode(&buf[1..len]).unwrap();
    assert_eq!(decoded.ascent, 800);
    assert_eq!(decoded.descent, -200);

    // Test EnsureGlyphs
    let req = EnsureGlyphs {
        face_id: ThingId::from_u64(0xABCD),
        px_size: 24,
        glyph_ids: vec![72, 101, 108, 108, 111], // "Hello"
    };
    let mut buf = [0u8; 256];
    let len = req.encode(&mut buf).unwrap();
    let decoded = EnsureGlyphs::decode(&buf[1..len]).unwrap();
    assert_eq!(decoded.face_id.to_u64_lossy(), 0xABCD);
    assert_eq!(decoded.px_size, 24);
    assert_eq!(decoded.glyph_ids, vec![72, 101, 108, 108, 111]);

    // Test GlyphPlacement
    let placement = GlyphPlacement {
        glyph_id: 72,
        x: 0,
        y: 0,
        w: 10,
        h: 14,
        bearing_x: 1,
        bearing_y: 12,
        advance: 11,
    };
    let mut buf = [0u8; 32];
    let len = placement.encode(&mut buf).unwrap();
    let decoded = GlyphPlacement::decode(&buf[..len]).unwrap();
    assert_eq!(decoded.glyph_id, 72);
    assert_eq!(decoded.w, 10);
    assert_eq!(decoded.advance, 11);
}
