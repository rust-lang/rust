use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use ucd_parse::Codepoints;

mod case_mapping;
mod raw_emitter;
mod unicode_download;

use raw_emitter::{emit_codepoints, RawEmitter};

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
            "e.g. {} src/libcore/unicode/unicode_data.rs",
            std::env::args().nth(0).unwrap_or_default()
        );
        std::process::exit(1);
    });

    let unicode_data = load_data();
    let ranges_by_property = &unicode_data.ranges;

    let mut total_bytes = 0;
    let mut modules = Vec::new();
    for (property, ranges) in ranges_by_property {
        let datapoints = ranges.iter().map(|r| r.end - r.start).sum::<u32>();
        let mut emitter = RawEmitter::new();
        emit_codepoints(&mut emitter, &ranges);

        modules.push((property.to_lowercase().to_string(), emitter.file));
        println!("{:15}: {} bytes, {} codepoints", property, emitter.bytes_used, datapoints,);
        total_bytes += emitter.bytes_used;
    }

    let mut table_file = String::new();

    table_file.push_str(
        "///! This file is generated by src/tools/unicode-table-generator; do not edit manually!\n",
    );

    table_file.push_str("use super::range_search;\n\n");

    table_file.push_str(&version());

    table_file.push('\n');

    modules.push((String::from("conversions"), case_mapping::generate_case_mapping(&unicode_data)));

    for (name, contents) in modules {
        table_file.push_str("#[rustfmt::skip]\n");
        table_file.push_str(&format!("pub mod {} {{\n", name));
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

    println!("Total table sizes: {} bytes", total_bytes);
}

fn version() -> String {
    let mut out = String::new();
    out.push_str("pub const UNICODE_VERSION: (u32, u32, u32) = ");

    let readme =
        std::fs::read_to_string(std::path::Path::new(UNICODE_DIRECTORY).join("ReadMe.txt"))
            .unwrap();

    let prefix = "for Version ";
    let start = readme.find(prefix).unwrap() + prefix.len();
    let end = readme.find(" of the Unicode Standard.").unwrap();
    let version =
        readme[start..end].split('.').map(|v| v.parse::<u32>().expect(&v)).collect::<Vec<_>>();
    let [major, minor, micro] = [version[0], version[1], version[2]];

    out.push_str(&format!("({}, {}, {});\n", major, minor, micro));
    out
}

fn fmt_list<V: std::fmt::Debug>(values: impl IntoIterator<Item = V>) -> String {
    let pieces = values.into_iter().map(|b| format!("{:?}, ", b)).collect::<Vec<_>>();
    let mut out = String::new();
    let mut line = format!("\n    ");
    for piece in pieces {
        if line.len() + piece.len() < 98 {
            line.push_str(&piece);
        } else {
            out.push_str(line.trim_end());
            out.push('\n');
            line = format!("    {}", piece);
        }
    }
    out.push_str(line.trim_end());
    out.push('\n');
    out
}

fn merge_ranges(ranges: &mut Vec<Range<u32>>) {
    loop {
        let mut new_ranges = Vec::new();
        let mut idx_iter = 0..(ranges.len() - 1);
        while let Some(idx) = idx_iter.next() {
            let cur = ranges[idx].clone();
            let next = ranges[idx + 1].clone();
            if cur.end == next.start {
                let _ = idx_iter.next(); // skip next as we're merging it in
                new_ranges.push(cur.start..next.end);
            } else {
                new_ranges.push(cur);
            }
        }
        new_ranges.push(ranges.last().unwrap().clone());
        if new_ranges.len() == ranges.len() {
            *ranges = new_ranges;
            break;
        } else {
            *ranges = new_ranges;
        }
    }
}
