//! This implements the core logic of the compression scheme used to compactly
//! encode Unicode properties.
//!
//! We have two primary goals with the encoding: we want to be compact, because
//! these tables often end up in ~every Rust program (especially the
//! grapheme_extend table, used for str debugging), including those for embedded
//! targets (where space is important). We also want to be relatively fast,
//! though this is more of a nice to have rather than a key design constraint.
//! It is expected that libraries/applications which are performance-sensitive
//! to Unicode property lookups are extremely rare, and those that care may find
//! the tradeoff of the raw bitsets worth it. For most applications, a
//! relatively fast but much smaller (and as such less cache-impacting, etc.)
//! data set is likely preferable.
//!
//! We have two separate encoding schemes: a skiplist-like approach, and a
//! compressed bitset. The datasets we consider mostly use the skiplist (it's
//! smaller) but the lowercase and uppercase sets are sufficiently sparse for
//! the bitset to be worthwhile -- for those sets the bitset is a 2x size win.
//! Since the bitset is also faster, this seems an obvious choice. (As a
//! historical note, the bitset was also the prior implementation, so its
//! relative complexity had already been paid).
//!
//! ## The bitset
//!
//! The primary idea is that we 'flatten' the Unicode ranges into an enormous
//! bitset. To represent any arbitrary codepoint in a raw bitset, we would need
//! over 17 kilobytes of data per character set -- way too much for our
//! purposes.
//!
//! First, the raw bitset (one bit for every valid `char`, from 0 to 0x10FFFF,
//! not skipping the small 'gap') is associated into words (u64) and
//! deduplicated. On random data, this would be useless; on our data, this is
//! incredibly beneficial -- our data sets have (far) less than 256 unique
//! words.
//!
//! This gives us an array that maps `u8 -> word`; the current algorithm does
//! not handle the case of more than 256 unique words, but we are relatively far
//! from coming that close.
//!
//! With that scheme, we now have a single byte for every 64 codepoints.
//!
//! We further chunk these by some constant N (between 1 and 64 per group,
//! dynamically chosen for smallest size), and again deduplicate and store in an
//! array (u8 -> [u8; N]).
//!
//! The bytes of this array map into the words from the bitset above, but we
//! apply another trick here: some of these words are similar enough that they
//! can be represented by some function of another word. The particular
//! functions chosen are rotation, inversion, and shifting (right).
//!
//! ## The skiplist
//!
//! The skip list arose out of the desire for an even smaller encoding than the
//! bitset -- and was the answer to the question "what is the smallest
//! representation we can imagine?". However, it is not necessarily the
//! smallest, and if you have a better proposal, please do suggest it!
//!
//! This is a relatively straightforward encoding. First, we break up all the
//! ranges in the input data into offsets from each other, essentially a gap
//! encoding. In practice, most gaps are small -- less than u8::MAX -- so we
//! store those directly. We make use of the larger gaps (which are nicely
//! interspersed already) throughout the dataset to index this data set.
//!
//! In particular, each run of small gaps (terminating in a large gap) is
//! indexed in a separate dataset. That data set stores an index into the
//! primary offset list and a prefix sum of that offset list. These are packed
//! into a single u32 (11 bits for the offset, 21 bits for the prefix sum).
//!
//! Lookup proceeds via a binary search in the index and then a straightforward
//! linear scan (adding up the offsets) until we reach the needle, and then the
//! index of that offset is utilized as the answer to whether we're in the set
//! or not.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use ucd_parse::Codepoints;

mod cascading_map;
mod case_mapping;
mod raw_emitter;
mod skiplist;
mod unicode_download;

use raw_emitter::{emit_codepoints, emit_whitespace, RawEmitter};

static PROPERTIES: &[&str] = &[
    "Alphabetic",
    "Lowercase",
    "Uppercase",
    "Cased",
    "Case_Ignorable",
    "Grapheme_Extend",
    "White_Space",
    "Cc",
    "N",
];

struct UnicodeData {
    ranges: Vec<(&'static str, Vec<Range<u32>>)>,
    to_upper: BTreeMap<u32, (u32, u32, u32)>,
    to_lower: BTreeMap<u32, (u32, u32, u32)>,
}

fn to_mapping(origin: u32, codepoints: Vec<ucd_parse::Codepoint>) -> Option<(u32, u32, u32)> {
    let mut a = None;
    let mut b = None;
    let mut c = None;

    for codepoint in codepoints {
        if origin == codepoint.value() {
            return None;
        }

        if a.is_none() {
            a = Some(codepoint.value());
        } else if b.is_none() {
            b = Some(codepoint.value());
        } else if c.is_none() {
            c = Some(codepoint.value());
        } else {
            panic!("more than 3 mapped codepoints")
        }
    }

    Some((a.unwrap(), b.unwrap_or(0), c.unwrap_or(0)))
}

static UNICODE_DIRECTORY: &str = "unicode-downloads";

fn load_data() -> UnicodeData {
    unicode_download::fetch_latest();

    let mut properties = HashMap::new();
    for row in ucd_parse::parse::<_, ucd_parse::CoreProperty>(&UNICODE_DIRECTORY).unwrap() {
        if let Some(name) = PROPERTIES.iter().find(|prop| **prop == row.property.as_str()) {
            properties.entry(*name).or_insert_with(Vec::new).push(row.codepoints);
        }
    }
    for row in ucd_parse::parse::<_, ucd_parse::Property>(&UNICODE_DIRECTORY).unwrap() {
        if let Some(name) = PROPERTIES.iter().find(|prop| **prop == row.property.as_str()) {
            properties.entry(*name).or_insert_with(Vec::new).push(row.codepoints);
        }
    }

    let mut to_lower = BTreeMap::new();
    let mut to_upper = BTreeMap::new();
    for row in ucd_parse::UnicodeDataExpander::new(
        ucd_parse::parse::<_, ucd_parse::UnicodeData>(&UNICODE_DIRECTORY).unwrap(),
    ) {
        let general_category = if ["Nd", "Nl", "No"].contains(&row.general_category.as_str()) {
            "N"
        } else {
            row.general_category.as_str()
        };
        if let Some(name) = PROPERTIES.iter().find(|prop| **prop == general_category) {
            properties
                .entry(*name)
                .or_insert_with(Vec::new)
                .push(Codepoints::Single(row.codepoint));
        }

        if let Some(mapped) = row.simple_lowercase_mapping {
            if mapped != row.codepoint {
                to_lower.insert(row.codepoint.value(), (mapped.value(), 0, 0));
            }
        }
        if let Some(mapped) = row.simple_uppercase_mapping {
            if mapped != row.codepoint {
                to_upper.insert(row.codepoint.value(), (mapped.value(), 0, 0));
            }
        }
    }

    for row in ucd_parse::parse::<_, ucd_parse::SpecialCaseMapping>(&UNICODE_DIRECTORY).unwrap() {
        if !row.conditions.is_empty() {
            // Skip conditional case mappings
            continue;
        }

        let key = row.codepoint.value();
        if let Some(lower) = to_mapping(key, row.lowercase) {
            to_lower.insert(key, lower);
        }
        if let Some(upper) = to_mapping(key, row.uppercase) {
            to_upper.insert(key, upper);
        }
    }

    let mut properties: HashMap<&'static str, Vec<Range<u32>>> = properties
        .into_iter()
        .map(|(k, v)| {
            (
                k,
                v.into_iter()
                    .flat_map(|codepoints| match codepoints {
                        Codepoints::Single(c) => c
                            .scalar()
                            .map(|ch| (ch as u32..ch as u32 + 1))
                            .into_iter()
                            .collect::<Vec<_>>(),
                        Codepoints::Range(c) => c
                            .into_iter()
                            .flat_map(|c| c.scalar().map(|ch| (ch as u32..ch as u32 + 1)))
                            .collect::<Vec<_>>(),
                    })
                    .collect::<Vec<Range<u32>>>(),
            )
        })
        .collect();

    for ranges in properties.values_mut() {
        merge_ranges(ranges);
    }

    let mut properties = properties.into_iter().collect::<Vec<_>>();
    properties.sort_by_key(|p| p.0);
    UnicodeData { ranges: properties, to_lower, to_upper }
}

fn main() {
    let write_location = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Must provide path to write unicode tables to");
        eprintln!(
            "e.g. {} library/core/src/unicode/unicode_data.rs",
            std::env::args().next().unwrap_or_default()
        );
        std::process::exit(1);
    });

    // Optional test path, which is a Rust source file testing that the unicode
    // property lookups are correct.
    let test_path = std::env::args().nth(2);

    let unicode_data = load_data();
    let ranges_by_property = &unicode_data.ranges;

    if let Some(path) = test_path {
        std::fs::write(&path, generate_tests(&write_location, &ranges_by_property)).unwrap();
    }

    let mut total_bytes = 0;
    let mut modules = Vec::new();
    for (property, ranges) in ranges_by_property {
        let datapoints = ranges.iter().map(|r| r.end - r.start).sum::<u32>();

        let mut emitter = RawEmitter::new();
        if property == &"White_Space" {
            emit_whitespace(&mut emitter, &ranges);
        } else {
            emit_codepoints(&mut emitter, &ranges);
        }

        modules.push((property.to_lowercase().to_string(), emitter.file));
        println!(
            "{:15}: {} bytes, {} codepoints in {} ranges ({} - {}) using {}",
            property,
            emitter.bytes_used,
            datapoints,
            ranges.len(),
            ranges.first().unwrap().start,
            ranges.last().unwrap().end,
            emitter.desc,
        );
        total_bytes += emitter.bytes_used;
    }

    let mut table_file = String::new();

    table_file.push_str(
        "///! This file is generated by src/tools/unicode-table-generator; do not edit manually!\n",
    );

    // Include the range search function
    table_file.push('\n');
    table_file.push_str(include_str!("range_search.rs"));
    table_file.push('\n');

    table_file.push_str(&version());

    table_file.push('\n');

    modules.push((String::from("conversions"), case_mapping::generate_case_mapping(&unicode_data)));

    for (name, contents) in modules {
        table_file.push_str("#[rustfmt::skip]\n");
        table_file.push_str(&format!("pub mod {name} {{\n"));
        for line in contents.lines() {
            if !line.trim().is_empty() {
                table_file.push_str("    ");
                table_file.push_str(&line);
            }
            table_file.push('\n');
        }
        table_file.push_str("}\n\n");
    }

    std::fs::write(&write_location, format!("{}\n", table_file.trim_end())).unwrap();

    println!("Total table sizes: {total_bytes} bytes");
}

fn version() -> String {
    let mut out = String::new();
    out.push_str("pub const UNICODE_VERSION: (u8, u8, u8) = ");

    let readme =
        std::fs::read_to_string(std::path::Path::new(UNICODE_DIRECTORY).join("ReadMe.txt"))
            .unwrap();

    let prefix = "for Version ";
    let start = readme.find(prefix).unwrap() + prefix.len();
    let end = readme.find(" of the Unicode Standard.").unwrap();
    let version =
        readme[start..end].split('.').map(|v| v.parse::<u32>().expect(&v)).collect::<Vec<_>>();
    let [major, minor, micro] = [version[0], version[1], version[2]];

    out.push_str(&format!("({major}, {minor}, {micro});\n"));
    out
}

fn fmt_list<V: std::fmt::Debug>(values: impl IntoIterator<Item = V>) -> String {
    let pieces = values.into_iter().map(|b| format!("{:?}, ", b)).collect::<Vec<_>>();
    let mut out = String::new();
    let mut line = String::from("\n    ");
    for piece in pieces {
        if line.len() + piece.len() < 98 {
            line.push_str(&piece);
        } else {
            out.push_str(line.trim_end());
            out.push('\n');
            line = format!("    {piece}");
        }
    }
    out.push_str(line.trim_end());
    out.push('\n');
    out
}

fn generate_tests(data_path: &str, ranges: &[(&str, Vec<Range<u32>>)]) -> String {
    let mut s = String::new();
    s.push_str("#![allow(incomplete_features, unused)]\n");
    s.push_str("#![feature(const_generics)]\n\n");
    s.push_str("\n#[allow(unused)]\nuse std::hint;\n");
    s.push_str(&format!("#[path = \"{data_path}\"]\n"));
    s.push_str("mod unicode_data;\n\n");

    s.push_str("\nfn main() {\n");

    for (property, ranges) in ranges {
        s.push_str(&format!(r#"    println!("Testing {}");"#, property));
        s.push('\n');
        s.push_str(&format!("    {}_true();\n", property.to_lowercase()));
        s.push_str(&format!("    {}_false();\n", property.to_lowercase()));
        let mut is_true = Vec::new();
        let mut is_false = Vec::new();
        for ch_num in 0..(std::char::MAX as u32) {
            if std::char::from_u32(ch_num).is_none() {
                continue;
            }
            if ranges.iter().any(|r| r.contains(&ch_num)) {
                is_true.push(ch_num);
            } else {
                is_false.push(ch_num);
            }
        }

        s.push_str(&format!("    fn {}_true() {{\n", property.to_lowercase()));
        generate_asserts(&mut s, property, &is_true, true);
        s.push_str("    }\n\n");
        s.push_str(&format!("    fn {}_false() {{\n", property.to_lowercase()));
        generate_asserts(&mut s, property, &is_false, false);
        s.push_str("    }\n\n");
    }

    s.push_str("}");
    s
}

fn generate_asserts(s: &mut String, property: &str, points: &[u32], truthy: bool) {
    for range in ranges_from_set(points) {
        if range.end == range.start + 1 {
            s.push_str(&format!(
                "        assert!({}unicode_data::{}::lookup({:?}), \"{}\");\n",
                if truthy { "" } else { "!" },
                property.to_lowercase(),
                std::char::from_u32(range.start).unwrap(),
                range.start,
            ));
        } else {
            s.push_str(&format!("        for chn in {:?}u32 {{\n", range));
            s.push_str(&format!(
                "            assert!({}unicode_data::{}::lookup(std::char::from_u32(chn).unwrap()), \"{{:?}}\", chn);\n",
                if truthy { "" } else { "!" },
                property.to_lowercase(),
            ));
            s.push_str("        }\n");
        }
    }
}

fn ranges_from_set(set: &[u32]) -> Vec<Range<u32>> {
    let mut ranges = set.iter().map(|e| (*e)..(*e + 1)).collect::<Vec<Range<u32>>>();
    merge_ranges(&mut ranges);
    ranges
}

fn merge_ranges(ranges: &mut Vec<Range<u32>>) {
    loop {
        let mut new_ranges = Vec::new();
        let mut idx_iter = 0..(ranges.len() - 1);
        let mut should_insert_last = true;
        while let Some(idx) = idx_iter.next() {
            let cur = ranges[idx].clone();
            let next = ranges[idx + 1].clone();
            if cur.end == next.start {
                if idx_iter.next().is_none() {
                    // We're merging the last element
                    should_insert_last = false;
                }
                new_ranges.push(cur.start..next.end);
            } else {
                // We're *not* merging the last element
                should_insert_last = true;
                new_ranges.push(cur);
            }
        }
        if should_insert_last {
            new_ranges.push(ranges.last().unwrap().clone());
        }
        if new_ranges.len() == ranges.len() {
            *ranges = new_ranges;
            break;
        } else {
            *ranges = new_ranges;
        }
    }

    let mut last_end = None;
    for range in ranges {
        if let Some(last) = last_end {
            assert!(range.start > last, "{:?}", range);
        }
        last_end = Some(range.end);
    }
}
