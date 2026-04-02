//! Generates lookup tables (LUTs) for "case-mapping": mapping a `char` to its
//! uppercase or lowercase equivalent(s).
//!
//! # Lookup algorithm
//!
//! The LUT is split into two levels:
//!
//! The 1st level LUT is simply an array of `L2Lut`s. The correct `L2Lut` is
//! found by splitting the input `codepoint` into an `(input_high, input_low)`
//! pair (Unicode plane and lower 16 bits, respectively), and using `input_high`
//! as the index.
//!
//! The 2nd level LUT is further split into two separate tables:
//! - `singles`, for characters that map to a single codepoint.
//! - `multis`, for characters that map to 2 or 3 codepoints (eg `ß` -> `SS`).
//!
//! `singles` is a sorted slice of `(input_low, output_delta)` pairs. The correct
//! pair is found by binary search. The final output codepoint is reconstructed
//! with `input_high << 16 | (input_low + output_delta)`.
//!
//! `multis` is a sorted slice of `(input_low, [output_low1, output_low2,
//! output_low3])` pairs. The correct pair is found by binary search. The final
//! output codepoints are reconstructed with `input_high << 16 | output_low`.
//!
//! # Size optimizations
//!
//! Splitting the LUT into `L1Lut` and `L2Lut` means we do not need to store the
//! upper 16 bits of every entry in the `L2Lut`, since every character stays
//! within the same plane when case-mapped. This cuts the total size of the LUT
//! in half.
//!
//! Storing the delta between the input codepoint and output codepoint in
//! `singles` allows us to combine contiguous ranges which all map to codepoints
//! the same distance away. For example, all of the accented letters in Latin-1
//! that map to a single codepoint can be case-mapped by adding or subtracting
//! 32.
//!
//! We also combine adjacent entries which only differ by 2, and indicate by
//! setting the `parity` bit in the `Range` struct. An input codepoint belongs
//! to the `Range` if `input_low & range.parity == range.start & range.parity`.
//! This allows us to combine even more entries, without increasing the size of
//! each entry.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::RangeInclusive;

use crate::fmt_helpers::Hex;
use crate::{UnicodeData, fmt_list};

pub(crate) fn generate_case_mapping(data: &UnicodeData) -> (String, [(String, usize); 4]) {
    let mut file = String::new();

    file.push_str("\n\n");
    file.push_str(HEADER.trim_start());
    file.push('\n');
    let (lower_tables, lower_desc, lower_size) = generate_tables("LOWERCASE", &data.to_lower);
    file.push_str(&lower_tables);
    file.push_str("\n\n");
    let (upper_tables, upper_desc, upper_size) = generate_tables("UPPERCASE", &data.to_upper);
    file.push_str(&upper_tables);
    file.push_str("\n\n");
    let (title_tables, title_desc, title_size) = generate_tables("TITLECASE", &data.to_title);
    file.push_str(&title_tables);
    file.push_str("\n\n");
    let (casefold_tables, casefold_desc, casefold_size) =
        generate_tables("CASEFOLD", &data.to_casefold);
    file.push_str(&casefold_tables);
    (
        file,
        [
            (lower_desc, lower_size),
            (upper_desc, upper_size),
            (title_desc, title_size),
            (casefold_desc, casefold_size),
        ],
    )
}

// So far, only planes 0 and 1 (Basic Multilingual Plane and Supplementary
// Multilingual Plane, respectively) have case mappings. If that changes, increment this.
const NUM_PLANES: u16 = 2;

/// Split a codepoint into its plane (upper 16 bits) and position within the plane (lower 16 bits).
fn deconstruct(c: u32) -> (u16, u16) {
    let high = (c >> 16) as u16;
    assert!(high <= NUM_PLANES);

    let low = c as u16;
    (high, low)
}

#[derive(Debug, Clone)]
struct L1Lut {
    /// Keyed by bits 16..=31 of the code point. So far, only planes 0 and 1
    /// have case mappings, so we don't need to bother storing the plane numbers.
    l2_luts: [L2Lut; NUM_PLANES as usize],
}

impl Default for L1Lut {
    fn default() -> Self {
        Self { l2_luts: [L2Lut::default(), L2Lut::default()] }
    }
}

impl L1Lut {
    fn size(&self) -> usize {
        self.l2_luts.iter().map(|l2| l2.size()).sum()
    }
}

#[derive(Default, Clone)]
struct L2Lut {
    /// Keyed by bits 0..=15 of the code point, see `Range` struct for how
    /// membership is tested.
    singles: Vec<(Range, i16)>,

    /// Keyed by bits 0..=15 of the code point, value is the lower 16-bits of the 2 or 3 code points that the char expands to.
    multis: Vec<(Hex<u16>, [Hex<u16>; 3])>,
}

/// A compact encoding of a `Range<u16>` in only 4 bytes.
/// If `parity` is false, the range represents code points `start..=start+len`,
/// with step 1.
/// If `parity` is true, the range represents code points `start..=start+len`,
/// with step 2.
/// Membership can be tested with `range.start() <= codepoint && codepoint <=
/// range.end() && (codepoint & parity == start & parity)`.
#[derive(Copy, Clone)]
struct Range {
    start: u16,
    len: u8,
    parity: bool,
    // 7 bits left for any more flags we may want in the future.
}

impl Range {
    const fn new(range: RangeInclusive<u16>, parity: bool) -> Self {
        let start = *range.start();
        let end = *range.end();
        assert!(start <= end);

        let len = end - start;
        assert!(len <= 255);

        Self { start, len: len as u8, parity }
    }

    const fn singleton(start: u16) -> Self {
        Self::new(start..=start, false)
    }

    const fn start(&self) -> u16 {
        self.start
    }

    const fn end(&self) -> u16 {
        self.start + self.len as u16
    }

    const fn len(&self) -> u16 {
        self.len as u16 + 1
    }
}

impl fmt::Debug for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len() == 1 {
            return write!(f, "Range::singleton({})", Hex(self.start));
        }

        let range = Hex(self.start)..=Hex(self.end());
        match self.parity {
            false => write!(f, "Range::step_by_1({range:?})"),
            true => write!(f, "Range::step_by_2({range:?})"),
        }
    }
}

impl L2Lut {
    fn size(&self) -> usize {
        size_of_val(self.singles.as_slice()) + size_of_val(self.multis.as_slice())
    }
}

impl fmt::Debug for L2Lut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let singles = self.singles.as_slice();
        let singles = fmt::from_fn(|f| {
            write!(f, "&[ // {} entries, {} bytes", singles.len(), size_of_val(singles))?;
            write!(f, "{}]", fmt_list(singles))
        });

        let multis = self.multis.as_slice();
        let multis = fmt::from_fn(|f| {
            write!(f, "&[ // {} entries, {} bytes", multis.len(), size_of_val(multis))?;
            write!(f, "{}]", fmt_list(multis))
        });

        f.debug_struct("L2Lut").field("singles", &singles).field("multis", &multis).finish()
    }
}

fn generate_tables(case: &str, data: &BTreeMap<u32, [u32; 3]>) -> (String, String, usize) {
    let mut l1_lut = L1Lut::default();

    for (&input, &output) in data.iter() {
        assert!(input > 0x7f);

        let (input_high, input_low) = deconstruct(input);
        let l2_lut = &mut l1_lut.l2_luts[input_high as usize];

        match output {
            [output, 0, 0] => {
                let (output_high, output_low) = deconstruct(output);
                assert_eq!(
                    output_high, input_high,
                    "Case-mapping a character should not change its plane"
                );
                let delta = output_low.wrapping_sub(input_low).cast_signed();
                let range = Range::singleton(input_low);
                l2_lut.singles.push((range, delta));
            }
            _ => {
                let output_lows = output.map(|output| {
                    let (output_high, output_low) = deconstruct(output);
                    assert_eq!(
                        output_high, input_high,
                        "Case-mapping a character should not change its plane"
                    );
                    Hex(output_low)
                });
                l2_lut.multis.push((Hex(input_low), output_lows));
            }
        }
    }

    // Combine adjacent contiguous ranges which apply the same delta.
    // eg `(0x1234..=0x1234, +1), (0x1235..=0x1235, +1), (0x1236..=0x1236, +1)`
    // becomes `(Range::step_by_1(0x1234..=0x1236), +1)`.
    for l2_lut in &mut l1_lut.l2_luts {
        l2_lut.singles = l2_lut
            .singles
            .chunk_by(|(left_range, left_delta), (right_range, right_delta)| {
                left_delta == right_delta && right_range.end() - left_range.start() == 1
            })
            .map(|chunk| {
                let &(first_range, _delta) = chunk.first().unwrap();
                let &(last_range, delta) = chunk.last().unwrap();
                (Range::new(first_range.start()..=last_range.end(), false), delta)
            })
            .collect();
    }

    // Combine adjacent ranges which differ by 2, since we can encode that with the parity bit in the Range struct.
    // eg `(0x1234..=0x1234, +1), (0x1236..=0x1236, +1), (0x1238..=0x1238, +1)`
    // becomes `(Range::step_by_2(0x1234..=0x1238), +1)`.
    for l2_lut in &mut l1_lut.l2_luts {
        l2_lut.singles = l2_lut
            .singles
            .chunk_by(|(left_range, left_delta), (right_range, right_delta)| {
                left_delta == right_delta
                    && left_range.len() == 1
                    && right_range.len() == 1
                    && right_range.end() - left_range.start() == 2
            })
            .map(|chunk| {
                let &(first_range, _delta) = chunk.first().unwrap();
                let &(last_range, delta) = chunk.last().unwrap();
                let parity = chunk.len() > 1;
                (Range::new(first_range.start()..=last_range.end(), parity), delta)
            })
            .collect();
    }

    let size = l1_lut.size();
    let num_ranges =
        l1_lut.l2_luts.iter().map(|l2| l2.singles.len() + l2.multis.len()).sum::<usize>();
    let table = format!("static {case}_LUT: L1Lut = {l1_lut:#?};");
    let desc = format!(
        "{:6} codepoints in {:3} ranges (U+{:06X} - U+{:06X}) using 2-level LUT",
        data.len(),
        num_ranges,
        data.first_key_value().map(|(&k, _)| k).unwrap(),
        data.last_key_value().map(|(&k, _)| k).unwrap(),
    );
    (table, desc, size)
}

static HEADER: &str = r"
use crate::ops::RangeInclusive;

struct L1Lut {
    l2_luts: [L2Lut; 2],
}

struct L2Lut {
    singles: &'static [(Range, i16)],
    multis: &'static [(u16, [u16; 3])],
}

#[derive(Copy, Clone)]
struct Range {
    start: u16,
    len: u8,
    parity: bool,
}

impl Range {
    const fn new(range: RangeInclusive<u16>, parity: bool) -> Self {
        let start = *range.start();
        let end = *range.end();
        assert!(start <= end);

        let len = end - start;
        assert!(len <= 255);

        Self { start, len: len as u8, parity }
    }

    const fn singleton(start: u16) -> Self {
        Self::new(start..=start, false)
    }

    const fn step_by_1(range: RangeInclusive<u16>) -> Self {
        Self::new(range, false)
    }

    const fn step_by_2(range: RangeInclusive<u16>) -> Self {
        Self::new(range, true)
    }

    const fn start(&self) -> u16 {
        self.start
    }

    const fn end(&self) -> u16 {
        self.start + self.len as u16
    }
}

fn deconstruct(c: char) -> (u16, u16) {
    let c = c as u32;
    let plane = (c >> 16) as u16;
    let low = c as u16;
    (plane, low)
}

unsafe fn reconstruct(plane: u16, low: u16) -> char {
    // SAFETY: The caller must ensure that the result is a valid `char`.
    unsafe { char::from_u32_unchecked(((plane as u32) << 16) | (low as u32)) }
}

fn lookup(input: char, l1_lut: &L1Lut) -> Option<[char; 3]> {
    let (input_high, input_low) = deconstruct(input);
    let Some(l2_lut) = l1_lut.l2_luts.get(input_high as usize) else {
        return None;
    };

    let idx = l2_lut.singles.binary_search_by(|(range, _)| {
        use crate::cmp::Ordering;

        if input_low < range.start() {
            Ordering::Greater
        } else if input_low > range.end() {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    });

    if let Ok(idx) = idx {
        // SAFETY: binary search guarantees that the index is in bounds.
        let &(range, output_delta) = unsafe { l2_lut.singles.get_unchecked(idx) };
        let mask = range.parity as u16;
        if input_low & mask == range.start() & mask {
            let output_low = input_low.wrapping_add_signed(output_delta);
            // SAFETY: Table data are guaranteed to be valid Unicode.
            let output = unsafe { reconstruct(input_high, output_low) };
            return Some([output, '\0', '\0']);
        }
    };

    if let Ok(idx) = l2_lut.multis.binary_search_by_key(&input_low, |&(p, _)| p) {
        // SAFETY: binary search guarantees that the index is in bounds.
        let &(_, output_lows) = unsafe { l2_lut.multis.get_unchecked(idx) };
        // SAFETY: Table data are guaranteed to be valid Unicode.
        let output = output_lows.map(|output_low| unsafe { reconstruct(input_high, output_low) });
        return Some(output);
    };

    None
}

pub fn to_lower(c: char) -> [char; 3] {
    // https://util.unicode.org/UnicodeJsps/list-unicodeset.jsp?a=[:Changes_When_Lowercased:]-[:ASCII:]&abb=on
    if c < '\u{C0}' {
        return [c.to_ascii_lowercase(), '\0', '\0'];
    }

    lookup(c, &LOWERCASE_LUT).unwrap_or([c, '\0', '\0'])
}

pub fn to_upper(c: char) -> [char; 3] {
    // https://util.unicode.org/UnicodeJsps/list-unicodeset.jsp?a=[:Changes_When_Uppercased:]-[:ASCII:]&abb=on
    if c < '\u{B5}' {
        return [c.to_ascii_uppercase(), '\0', '\0'];
    }

    lookup(c, &UPPERCASE_LUT).unwrap_or([c, '\0', '\0'])
}

pub fn to_title(c: char) -> [char; 3] {
    // https://util.unicode.org/UnicodeJsps/list-unicodeset.jsp?a=[:Changes_When_Titlecased:]-[:ASCII:]&abb=on
    if c < '\u{B5}' {
        return [c.to_ascii_uppercase(), '\0', '\0'];
    }

    lookup(c, &TITLECASE_LUT).or_else(|| lookup(c, &UPPERCASE_LUT)).unwrap_or([c, '\0', '\0'])
}

pub fn to_casefold(c: char) -> [char; 3] {
    // https://util.unicode.org/UnicodeJsps/list-unicodeset.jsp?a=[:Changes_When_Casefolded:]-[:ASCII:]&abb=on
    if c < '\u{B5}' {
        return [c.to_ascii_lowercase(), '\0', '\0'];
    }


    lookup(c, &CASEFOLD_LUT).unwrap_or_else(|| {
        // fall back to lowercase of uppercase

        let uppercase = lookup(c, &UPPERCASE_LUT).unwrap_or([c, '\0', '\0']);
        let mut final_result = to_lower(uppercase[0]);
        if uppercase[1] != '\0' {
            let lowercase_1 = to_lower(uppercase[1]);
            debug_assert_eq!(lowercase_1[2], '\0');

            // If, after updating the Unicode data
            // to a new Unicode version, the below
            // assertion starts to fail in tests,
            // delete it, and uncomment the
            // `if` condition and corresponding
            // `else` block below it.
            debug_assert_eq!(final_result[1], '\0');
            //if final_result[1] == '\0' {

            final_result[1] = lowercase_1[0];

            if uppercase[2] != '\0' {
                debug_assert_eq!(lowercase_1[1], '\0');
                let lowercase_2 = to_lower(uppercase[2]);
                debug_assert_eq!(lowercase_2[1], '\0');
                debug_assert_eq!(lowercase_2[2], '\0');
                final_result[2] = lowercase_2[0];
            } else {
                // If, after updating the Unicode data
                // to a new Unicode version, the below
                // assertion starts to fail in tests,
                // delete it and uncomment the line
                // below it.
                debug_assert_eq!(lowercase_1[1], '\0');
                //final_result[2] = lowercase_1[1];
            }

            /*} else {
                final_result[2] = lowercase_1[0];
                debug_assert_eq!(lowercase_1[1], '\0');
                debug_assert_eq!(uppercase[2], '\0')
            }*/
        }
        final_result
    })
}
";
