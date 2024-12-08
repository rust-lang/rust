use itertools::EitherOrBoth;
use itertools::Itertools;

/// Iterator which breaks an identifier into various [VersionChunk]s.
struct VersionChunkIter<'a> {
    ident: &'a str,
    start: usize,
}

impl<'a> VersionChunkIter<'a> {
    pub(crate) fn new(ident: &'a str) -> Self {
        Self { ident, start: 0 }
    }

    fn parse_numeric_chunk(
        &mut self,
        mut chars: std::str::CharIndices<'a>,
    ) -> Option<VersionChunk<'a>> {
        let mut end = self.start;
        let mut is_end_of_chunk = false;

        while let Some((idx, c)) = chars.next() {
            end = self.start + idx;

            if c.is_ascii_digit() {
                continue;
            }

            is_end_of_chunk = true;
            break;
        }

        let source = if is_end_of_chunk {
            let value = &self.ident[self.start..end];
            self.start = end;
            value
        } else {
            let value = &self.ident[self.start..];
            self.start = self.ident.len();
            value
        };

        let zeros = source.chars().take_while(|c| *c == '0').count();
        let value = source.parse::<usize>().ok()?;

        Some(VersionChunk::Number {
            value,
            zeros,
            source,
        })
    }

    fn parse_str_chunk(
        &mut self,
        mut chars: std::str::CharIndices<'a>,
    ) -> Option<VersionChunk<'a>> {
        let mut end = self.start;
        let mut is_end_of_chunk = false;

        while let Some((idx, c)) = chars.next() {
            end = self.start + idx;

            if c == '_' {
                is_end_of_chunk = true;
                break;
            }

            if !c.is_numeric() {
                continue;
            }

            is_end_of_chunk = true;
            break;
        }

        let source = if is_end_of_chunk {
            let value = &self.ident[self.start..end];
            self.start = end;
            value
        } else {
            let value = &self.ident[self.start..];
            self.start = self.ident.len();
            value
        };

        Some(VersionChunk::Str(source))
    }
}

impl<'a> Iterator for VersionChunkIter<'a> {
    type Item = VersionChunk<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut chars = self.ident[self.start..].char_indices();
        let (_, next) = chars.next()?;

        if next == '_' {
            self.start = self.start + next.len_utf8();
            return Some(VersionChunk::Underscore);
        }

        if next.is_ascii_digit() {
            return self.parse_numeric_chunk(chars);
        }

        self.parse_str_chunk(chars)
    }
}

/// Represents a chunk in the version-sort algorithm
#[derive(Debug, PartialEq, Eq)]
enum VersionChunk<'a> {
    /// A single `_` in an identifier. Underscores are sorted before all other characters.
    Underscore,
    /// A &str chunk in the version sort.
    Str(&'a str),
    /// A numeric chunk in the version sort. Keeps track of the numeric value and leading zeros.
    Number {
        value: usize,
        zeros: usize,
        source: &'a str,
    },
}

/// Determine which side of the version-sort comparison had more leading zeros.
#[derive(Debug, PartialEq, Eq)]
enum MoreLeadingZeros {
    Left,
    Right,
    Equal,
}

/// Compare two identifiers based on the version sorting algorithm described in [the style guide]
///
/// [the style guide]: https://doc.rust-lang.org/nightly/style-guide/#sorting
pub(crate) fn version_sort(a: &str, b: &str) -> std::cmp::Ordering {
    let iter_a = VersionChunkIter::new(a);
    let iter_b = VersionChunkIter::new(b);
    let mut more_leading_zeros = MoreLeadingZeros::Equal;

    for either_or_both in iter_a.zip_longest(iter_b) {
        match either_or_both {
            EitherOrBoth::Left(_) => return std::cmp::Ordering::Greater,
            EitherOrBoth::Right(_) => return std::cmp::Ordering::Less,
            EitherOrBoth::Both(a, b) => match (a, b) {
                (VersionChunk::Underscore, VersionChunk::Underscore) => {
                    continue;
                }
                (VersionChunk::Underscore, _) => return std::cmp::Ordering::Less,
                (_, VersionChunk::Underscore) => return std::cmp::Ordering::Greater,
                (VersionChunk::Str(ca), VersionChunk::Str(cb))
                | (VersionChunk::Str(ca), VersionChunk::Number { source: cb, .. })
                | (VersionChunk::Number { source: ca, .. }, VersionChunk::Str(cb)) => {
                    match ca.cmp(&cb) {
                        std::cmp::Ordering::Equal => {
                            continue;
                        }
                        order @ _ => return order,
                    }
                }
                (
                    VersionChunk::Number {
                        value: va,
                        zeros: lza,
                        ..
                    },
                    VersionChunk::Number {
                        value: vb,
                        zeros: lzb,
                        ..
                    },
                ) => match va.cmp(&vb) {
                    std::cmp::Ordering::Equal => {
                        if lza == lzb {
                            continue;
                        }

                        if more_leading_zeros == MoreLeadingZeros::Equal && lza > lzb {
                            more_leading_zeros = MoreLeadingZeros::Left;
                        } else if more_leading_zeros == MoreLeadingZeros::Equal && lza < lzb {
                            more_leading_zeros = MoreLeadingZeros::Right;
                        }
                        continue;
                    }
                    order @ _ => return order,
                },
            },
        }
    }

    match more_leading_zeros {
        MoreLeadingZeros::Equal => std::cmp::Ordering::Equal,
        MoreLeadingZeros::Left => std::cmp::Ordering::Less,
        MoreLeadingZeros::Right => std::cmp::Ordering::Greater,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_chunks() {
        let mut iter = VersionChunkIter::new("x86_128");
        assert_eq!(iter.next(), Some(VersionChunk::Str("x")));
        assert_eq!(
            iter.next(),
            Some(VersionChunk::Number {
                value: 86,
                zeros: 0,
                source: "86"
            })
        );
        assert_eq!(iter.next(), Some(VersionChunk::Underscore));
        assert_eq!(
            iter.next(),
            Some(VersionChunk::Number {
                value: 128,
                zeros: 0,
                source: "128"
            })
        );
        assert_eq!(iter.next(), None);

        let mut iter = VersionChunkIter::new("w005s09t");
        assert_eq!(iter.next(), Some(VersionChunk::Str("w")));
        assert_eq!(
            iter.next(),
            Some(VersionChunk::Number {
                value: 5,
                zeros: 2,
                source: "005"
            })
        );
        assert_eq!(iter.next(), Some(VersionChunk::Str("s")));
        assert_eq!(
            iter.next(),
            Some(VersionChunk::Number {
                value: 9,
                zeros: 1,
                source: "09"
            })
        );
        assert_eq!(iter.next(), Some(VersionChunk::Str("t")));
        assert_eq!(iter.next(), None);

        let mut iter = VersionChunkIter::new("ZY_WX");
        assert_eq!(iter.next(), Some(VersionChunk::Str("ZY")));
        assert_eq!(iter.next(), Some(VersionChunk::Underscore));
        assert_eq!(iter.next(), Some(VersionChunk::Str("WX")));

        let mut iter = VersionChunkIter::new("_v1");
        assert_eq!(iter.next(), Some(VersionChunk::Underscore));
        assert_eq!(iter.next(), Some(VersionChunk::Str("v")));
        assert_eq!(
            iter.next(),
            Some(VersionChunk::Number {
                value: 1,
                zeros: 0,
                source: "1"
            })
        );

        let mut iter = VersionChunkIter::new("_1v");
        assert_eq!(iter.next(), Some(VersionChunk::Underscore));
        assert_eq!(
            iter.next(),
            Some(VersionChunk::Number {
                value: 1,
                zeros: 0,
                source: "1"
            })
        );
        assert_eq!(iter.next(), Some(VersionChunk::Str("v")));

        let mut iter = VersionChunkIter::new("v009");
        assert_eq!(iter.next(), Some(VersionChunk::Str("v")));
        assert_eq!(
            iter.next(),
            Some(VersionChunk::Number {
                value: 9,
                zeros: 2,
                source: "009"
            })
        );
    }

    #[test]
    fn test_version_sort() {
        let mut input = vec!["", "b", "a"];
        let expected = vec!["", "a", "b"];
        input.sort_by(|a, b| version_sort(a, b));
        assert_eq!(input, expected);

        let mut input = vec!["x7x", "xxx"];
        let expected = vec!["x7x", "xxx"];
        input.sort_by(|a, b| version_sort(a, b));
        assert_eq!(input, expected);

        let mut input = vec!["applesauce", "apple"];
        let expected = vec!["apple", "applesauce"];
        input.sort_by(|a, b| version_sort(a, b));
        assert_eq!(input, expected);

        let mut input = vec!["aaaaa", "aaa_a"];
        let expected = vec!["aaa_a", "aaaaa"];
        input.sort_by(|a, b| version_sort(a, b));
        assert_eq!(input, expected);

        let mut input = vec!["AAAAA", "AAA1A", "BBBBB", "BB_BB", "C3CCC"];
        let expected = vec!["AAA1A", "AAAAA", "BB_BB", "BBBBB", "C3CCC"];
        input.sort_by(|a, b| version_sort(a, b));
        assert_eq!(input, expected);

        let mut input = vec!["1_000_000", "1_010_001"];
        let expected = vec!["1_000_000", "1_010_001"];
        input.sort_by(|a, b| version_sort(a, b));
        assert_eq!(input, expected);

        let mut input = vec![
            "5", "50", "500", "5_000", "5_005", "5_050", "5_500", "50_000", "50_005", "50_050",
            "50_500",
        ];
        let expected = vec![
            "5", "5_000", "5_005", "5_050", "5_500", "50", "50_000", "50_005", "50_050", "50_500",
            "500",
        ];
        input.sort_by(|a, b| version_sort(a, b));
        assert_eq!(input, expected);

        let mut input = vec!["X86_64", "x86_64", "X86_128", "x86_128"];
        let expected = vec!["X86_64", "X86_128", "x86_64", "x86_128"];
        input.sort_by(|a, b| version_sort(a, b));
        assert_eq!(input, expected);

        let mut input = vec!["__", "_"];
        let expected = vec!["_", "__"];
        input.sort_by(|a, b| version_sort(a, b));
        assert_eq!(input, expected);

        let mut input = vec!["foo_", "foo"];
        let expected = vec!["foo", "foo_"];
        input.sort_by(|a, b| version_sort(a, b));
        assert_eq!(input, expected);

        let mut input = vec!["A", "AA", "B", "a", "aA", "aa", "b"];
        let expected = vec!["A", "AA", "B", "a", "aA", "aa", "b"];
        input.sort_by(|a, b| version_sort(a, b));
        assert_eq!(input, expected);

        let mut input = vec![
            "x86_128", "usize", "uz", "v000", "v00", "v0", "v0s", "v00t", "v0u", "v001", "v01",
            "v1", "v009", "x87", "zyxw", "_ZYXW", "_abcd", "A2", "ABCD", "Z_YXW", "ZY_XW", "ZY_XW",
            "ZYXW", "v09", "v9", "v010", "v10", "w005s09t", "w5s009t", "x64", "x86", "x86_32",
            "ua", "x86_64", "ZYXW_", "a1", "abcd", "u_zzz", "u8", "u16", "u32", "u64", "u128",
            "u256",
        ];
        let expected = vec![
            "_ZYXW", "_abcd", "A2", "ABCD", "Z_YXW", "ZY_XW", "ZY_XW", "ZYXW", "ZYXW_", "a1",
            "abcd", "u_zzz", "u8", "u16", "u32", "u64", "u128", "u256", "ua", "usize", "uz",
            "v000", "v00", "v0", "v0s", "v00t", "v0u", "v001", "v01", "v1", "v009", "v09", "v9",
            "v010", "v10", "w005s09t", "w5s009t", "x64", "x86", "x86_32", "x86_64", "x86_128",
            "x87", "zyxw",
        ];
        input.sort_by(|a, b| version_sort(a, b));
        assert_eq!(input, expected)
    }
}
