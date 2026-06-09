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

use std::collections::BTreeMap;
use std::fmt::Write;
use std::ops::Range;

use rustc_hash::{FxHashMap, FxHashSet};
use ucd_parse::{Codepoint, Codepoints};

mod cascading_map;
mod case_mapping;
mod fmt_helpers;
mod raw_emitter;
mod skiplist;
mod unicode_download;

use fmt_helpers::CharEscape;
use raw_emitter::{RawEmitter, emit_codepoints, emit_whitespace};

static PROPERTIES: &[&str] = &[
    "Alphabetic",
    "Lowercase",
    "Uppercase",
    "Case_Ignorable",
    "Grapheme_Extend",
    "White_Space",
    "N",
    "Lt",
];

struct UnicodeData {
    ranges: Vec<(&'static str, Vec<Range<u32>>)>,
    /// Only stores mappings that are not to self
    to_upper: BTreeMap<u32, [u32; 3]>,
    /// Only stores mappings that differ from `to_upper`
    to_title: BTreeMap<u32, [u32; 3]>,
    /// Only stores mappings that are not to self
    to_lower: BTreeMap<u32, [u32; 3]>,
    /// Only stores mappings that differ from
    /// `to_upper` followed by `to_lower`
    to_casefold: BTreeMap<u32, [u32; 3]>,
}

fn to_mapping(
    if_different_from: &[ucd_parse::Codepoint],
    codepoints: &[ucd_parse::Codepoint],
) -> Option<[u32; 3]> {
    if codepoints == if_different_from {
        return None;
    }

    let mut ret = [ucd_parse::Codepoint::default(); 3];
    ret[0..codepoints.len()].copy_from_slice(codepoints);
    Some(ret.map(ucd_parse::Codepoint::value))
}

static UNICODE_DIRECTORY: &str = "unicode-downloads";

fn load_data() -> UnicodeData {
    unicode_download::fetch_latest();

    let mut properties = FxHashMap::default();
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

    let [mut to_lower, mut to_upper, mut to_title, mut to_casefold] =
        [const { BTreeMap::new() }; 4];
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

        if let Some(mapped) = row.simple_lowercase_mapping
            && mapped != row.codepoint
        {
            to_lower.insert(row.codepoint.value(), [mapped.value(), 0, 0]);
        }
        if let Some(mapped) = row.simple_uppercase_mapping
            && mapped != row.codepoint
        {
            to_upper.insert(row.codepoint.value(), [mapped.value(), 0, 0]);
        }
        if let Some(mapped) = row.simple_titlecase_mapping
            && Some(mapped) != row.simple_uppercase_mapping
        {
            to_title.insert(row.codepoint.value(), [mapped.value(), 0, 0]);
        }
    }

    for row in ucd_parse::parse::<_, ucd_parse::SpecialCaseMapping>(&UNICODE_DIRECTORY).unwrap() {
        if !row.conditions.is_empty() {
            // Skip conditional case mappings
            continue;
        }

        let key = row.codepoint.value();
        if let Some(lower) = to_mapping(&[row.codepoint], &row.lowercase) {
            to_lower.insert(key, lower);
        }
        if let Some(upper) = to_mapping(&[row.codepoint], &row.uppercase) {
            to_upper.insert(key, upper);
        }
        if let Some(title) = to_mapping(&row.uppercase, &row.titlecase) {
            to_title.insert(key, title);
        }
    }

    fn get_mapping_from_btreemap<'a>(
        cp: Codepoint,
        map: &'a BTreeMap<u32, [u32; 3]>,
    ) -> Vec<Codepoint> {
        let mapping =
            map.get(&cp.value()).copied().map(|cs| cs.map(|c| Codepoint::from_u32(c).unwrap()));

        mapping
            .as_ref()
            .map(|cs| {
                let nul = Codepoint::from_u32(0).unwrap();
                if cs[1] == nul {
                    &cs[..1]
                } else if cs[2] == nul {
                    &cs[..2]
                } else {
                    &cs[..]
                }
            })
            .map_or_else(|| vec![cp], ToOwned::to_owned)
    }

    let mut nontrivial_casefold = FxHashSet::default();

    for row in ucd_parse::parse::<_, ucd_parse::CaseFold>(&UNICODE_DIRECTORY).unwrap() {
        use ucd_parse::{CaseStatus, Codepoint};
        if matches!(row.status, CaseStatus::Common | CaseStatus::Full) {
            let key = row.codepoint.value();
            nontrivial_casefold.insert(key);

            // We store case-fold data only for characters whose case-folding
            // differs from the lowercase of their uppercase.

            let lower_upper_mapping: Vec<Codepoint> =
                get_mapping_from_btreemap(row.codepoint, &to_upper)
                    .into_iter()
                    .flat_map(|cp| get_mapping_from_btreemap(cp, &to_lower))
                    .collect();

            if let Some(casefold) = to_mapping(&lower_upper_mapping, &row.mapping) {
                to_casefold.insert(key, casefold);
            }
        }
    }

    // Now, account for characters that remain unchanged by case-folding
    // (and are therefore omitted from `CaseFolding.txt`),
    // but yet differ from the lowercase of their uppercase.

    for c in '\0'..=char::MAX {
        let cnum: u32 = c.into();
        if !nontrivial_casefold.contains(&cnum) {
            let cp = Codepoint::from_u32(cnum).unwrap();

            use std::collections::btree_map::Entry;
            match to_casefold.entry(cnum) {
                Entry::Vacant(vacant_entry) => {
                    let lower_upper_mapping: Vec<Codepoint> =
                        get_mapping_from_btreemap(cp, &to_upper)
                            .into_iter()
                            .flat_map(|cp| get_mapping_from_btreemap(cp, &to_lower))
                            .collect();

                    if let Some(casefold) = to_mapping(&lower_upper_mapping, &[cp]) {
                        vacant_entry.insert(casefold);
                    }
                }
                Entry::Occupied(_) => {}
            }
        }
    }

    // Filter out ASCII codepoints.
    to_lower.retain(|&c, _| c > 0x7f);
    to_upper.retain(|&c, _| c > 0x7f);
    let mut properties: Vec<(&'static str, Vec<Range<u32>>)> = properties
        .into_iter()
        .map(|(prop, codepoints)| {
            let codepoints = codepoints
                .into_iter()
                .flatten()
                .flat_map(|cp| cp.scalar())
                .filter(|c| !c.is_ascii())
                .map(u32::from)
                .collect::<Vec<_>>();
            (prop, ranges_from_set(&codepoints))
        })
        .collect();

    properties.sort_by_key(|p| p.0);
    UnicodeData { ranges: properties, to_lower, to_title, to_upper, to_casefold }
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();

    if args.len() != 3 {
        eprintln!("Must provide paths to write unicode tables and tests to");
        eprintln!(
            "e.g. {} library/core/src/unicode/unicode_data.rs library/coretests/tests/unicode/test_data.rs",
            args[0]
        );
        std::process::exit(1);
    }

    let data_path = &args[1];
    let test_path = &args[2];

    let unicode_data = load_data();
    let ranges_by_property = &unicode_data.ranges;

    let mut table_file = String::new();
    table_file.push_str(
        "//! This file is generated by `./x run src/tools/unicode-table-generator`; do not edit manually!\n",
    );

    let mut total_bytes = 0;
    let mut modules = Vec::new();
    for (property, ranges) in ranges_by_property {
        let datapoints = ranges.iter().map(|r| r.end - r.start).sum::<u32>();

        let mut emitter = RawEmitter::new();
        if property == &"White_Space" {
            emit_whitespace(&mut emitter, ranges);
        } else {
            emit_codepoints(&mut emitter, ranges);
        }

        modules.push((property.to_lowercase().to_string(), emitter.file));
        table_file.push_str(&format!(
            "// {:16}: {:5} bytes, {:6} codepoints in {:3} ranges (U+{:06X} - U+{:06X}) using {}\n",
            property,
            emitter.bytes_used,
            datapoints,
            ranges.len(),
            ranges.first().unwrap().start,
            ranges.last().unwrap().end,
            emitter.desc,
        ));
        total_bytes += emitter.bytes_used;
    }
    let (conversions, sizes) = case_mapping::generate_case_mapping(&unicode_data);
    for (name, (desc, size)) in
        ["to_lower", "to_upper", "to_title", "to_casefold"].iter().zip(sizes)
    {
        table_file.push_str(&format!("// {:16}: {:5} bytes, {desc}\n", name, size,));
        total_bytes += size;
    }
    table_file.push_str(&format!("// {:16}: {:5} bytes\n", "Total", total_bytes));

    // Include the range search function
    table_file.push('\n');
    table_file.push_str(include_str!("range_search.rs"));
    table_file.push('\n');

    table_file.push_str(&version());

    table_file.push('\n');

    modules.push((String::from("conversions"), conversions));

    for (name, contents) in modules {
        table_file.push_str("#[rustfmt::skip]\n");
        table_file.push_str(&format!("pub mod {name} {{\n"));
        for line in contents.lines() {
            if !line.trim().is_empty() {
                table_file.push_str("    ");
                table_file.push_str(line);
            }
            table_file.push('\n');
        }
        table_file.push_str("}\n\n");
    }

    let test_file = generate_tests(&unicode_data);
    std::fs::write(&test_path, test_file).unwrap();
    std::fs::write(&data_path, table_file).unwrap();
    eprintln!("Unicode data was generated. Remember to run \"x fmt\"!");
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
        readme[start..end].split('.').map(|v| v.parse::<u32>().expect(v)).collect::<Vec<_>>();
    let [major, minor, micro] = [version[0], version[1], version[2]];

    out.push_str(&format!("({major}, {minor}, {micro});\n"));
    out
}

fn fmt_list<V: std::fmt::Debug>(values: impl IntoIterator<Item = V>) -> String {
    let pieces = values.into_iter().map(|b| format!("{b:?}, ")).collect::<Vec<_>>();
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

fn generate_tests(data: &UnicodeData) -> String {
    let mut out = String::from(
        "\
//! This file is generated by `./x run src/tools/unicode-table-generator`; do not edit manually!
// ignore-tidy-filelength

use std::ops::RangeInclusive;
",
    );
    for (property, ranges) in &data.ranges {
        let prop_upper = property.to_uppercase();
        let is_true = (char::MIN..=char::MAX)
            .filter(|c| !c.is_ascii())
            .map(u32::from)
            .filter(|c| ranges.iter().any(|r| r.contains(c)))
            .collect::<Vec<_>>();
        let is_true = ranges_from_set(&is_true)
            .into_iter()
            .map(|r| {
                let start = char::from_u32(r.start).unwrap();
                let end = char::from_u32(r.end - 1).unwrap();
                CharEscape(start)..=CharEscape(end)
            })
            .collect::<Vec<_>>();

        writeln!(
            out,
            r#"
#[rustfmt::skip]
pub(super) static {prop_upper}: &[RangeInclusive<char>; {is_true_len}] = &[{is_true}];
"#,
            is_true_len = is_true.len(),
            is_true = fmt_list(is_true),
        )
        .unwrap();
    }

    for (name, lut) in ["TO_LOWER", "TO_UPPER", "TO_TITLE", "TO_CASEFOLD"].iter().zip([
        &data.to_lower,
        &data.to_upper,
        &data.to_title,
        &data.to_casefold,
    ]) {
        let lut = lut
            .iter()
            .map(|(key, values)| {
                let key = char::from_u32(*key).unwrap();
                let values = values.map(|c| char::from_u32(c).unwrap());
                (CharEscape(key), values.map(CharEscape))
            })
            .collect::<Vec<_>>();

        writeln!(
            out,
            r#"
#[rustfmt::skip]
pub(super) static {name}: &[(char, [char; 3]); {len}] = &[{lut}];
"#,
            len = lut.len(),
            lut = fmt_list(lut),
        )
        .unwrap();
    }

    out
}

/// Group the elements of `set` into contigous ranges
fn ranges_from_set(set: &[u32]) -> Vec<Range<u32>> {
    set.chunk_by(|a, b| a + 1 == *b)
        .map(|chunk| {
            let start = *chunk.first().unwrap();
            let end = *chunk.last().unwrap();
            start..(end + 1)
        })
        .collect()
}
