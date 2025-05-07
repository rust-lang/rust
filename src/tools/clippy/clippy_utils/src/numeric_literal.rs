use rustc_ast::ast::{LitFloatType, LitIntType, LitKind};
use std::iter;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Radix {
    Binary,
    Octal,
    Decimal,
    Hexadecimal,
}

impl Radix {
    /// Returns a reasonable digit group size for this radix.
    #[must_use]
    fn suggest_grouping(self) -> usize {
        match self {
            Self::Binary | Self::Hexadecimal => 4,
            Self::Octal | Self::Decimal => 3,
        }
    }
}

/// A helper method to format numeric literals with digit grouping.
/// `lit` must be a valid numeric literal without suffix.
pub fn format(lit: &str, type_suffix: Option<&str>, float: bool) -> String {
    NumericLiteral::new(lit, type_suffix, float).format()
}

#[derive(Debug)]
pub struct NumericLiteral<'a> {
    /// Which radix the literal was represented in.
    pub radix: Radix,
    /// The radix prefix, if present.
    pub prefix: Option<&'a str>,

    /// The integer part of the number.
    pub integer: &'a str,
    /// The fraction part of the number.
    pub fraction: Option<&'a str>,
    /// The exponent separator (b'e' or b'E') including preceding underscore if present
    /// and the exponent part.
    pub exponent: Option<(&'a str, &'a str)>,

    /// The type suffix, including preceding underscore if present.
    pub suffix: Option<&'a str>,
}

impl<'a> NumericLiteral<'a> {
    pub fn from_lit_kind(src: &'a str, lit_kind: &LitKind) -> Option<NumericLiteral<'a>> {
        let unsigned_src = src.strip_prefix('-').map_or(src, |s| s);
        if lit_kind.is_numeric()
            && unsigned_src
                .trim_start()
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit())
        {
            let (unsuffixed, suffix) = split_suffix(src, lit_kind);
            let float = matches!(lit_kind, LitKind::Float(..));
            Some(NumericLiteral::new(unsuffixed, suffix, float))
        } else {
            None
        }
    }

    #[must_use]
    pub fn new(lit: &'a str, suffix: Option<&'a str>, float: bool) -> Self {
        let unsigned_lit = lit.trim_start_matches('-');
        // Determine delimiter for radix prefix, if present, and radix.
        let radix = if unsigned_lit.starts_with("0x") {
            Radix::Hexadecimal
        } else if unsigned_lit.starts_with("0b") {
            Radix::Binary
        } else if unsigned_lit.starts_with("0o") {
            Radix::Octal
        } else {
            Radix::Decimal
        };

        // Grab part of the literal after prefix, if present.
        let (prefix, mut sans_prefix) = if radix == Radix::Decimal {
            (None, lit)
        } else {
            let (p, s) = lit.split_at(2);
            (Some(p), s)
        };

        if suffix.is_some() && sans_prefix.ends_with('_') {
            // The '_' before the suffix isn't part of the digits
            sans_prefix = &sans_prefix[..sans_prefix.len() - 1];
        }

        let (integer, fraction, exponent) = Self::split_digit_parts(sans_prefix, float);

        Self {
            radix,
            prefix,
            integer,
            fraction,
            exponent,
            suffix,
        }
    }

    pub fn is_decimal(&self) -> bool {
        self.radix == Radix::Decimal
    }

    pub fn split_digit_parts(digits: &str, float: bool) -> (&str, Option<&str>, Option<(&str, &str)>) {
        let mut integer = digits;
        let mut fraction = None;
        let mut exponent = None;

        if float {
            for (i, c) in digits.char_indices() {
                match c {
                    '.' => {
                        integer = &digits[..i];
                        fraction = Some(&digits[i + 1..]);
                    },
                    'e' | 'E' => {
                        let exp_start = if digits[..i].ends_with('_') { i - 1 } else { i };

                        if integer.len() > exp_start {
                            integer = &digits[..exp_start];
                        } else {
                            fraction = Some(&digits[integer.len() + 1..exp_start]);
                        }
                        exponent = Some((&digits[exp_start..=i], &digits[i + 1..]));
                        break;
                    },
                    _ => {},
                }
            }
        }

        (integer, fraction, exponent)
    }

    /// Returns literal formatted in a sensible way.
    pub fn format(&self) -> String {
        let mut output = String::new();

        if let Some(prefix) = self.prefix {
            output.push_str(prefix);
        }

        let group_size = self.radix.suggest_grouping();

        Self::group_digits(
            &mut output,
            self.integer,
            group_size,
            true,
            self.radix == Radix::Hexadecimal,
        );

        if let Some(fraction) = self.fraction {
            output.push('.');
            Self::group_digits(&mut output, fraction, group_size, false, false);
        }

        if let Some((separator, exponent)) = self.exponent {
            if !exponent.is_empty() && exponent != "0" {
                output.push_str(separator);
                Self::group_digits(&mut output, exponent, group_size, true, false);
            } else if exponent == "0" && self.fraction.is_none() && self.suffix.is_none() {
                output.push_str(".0");
            }
        }

        if let Some(suffix) = self.suffix {
            if output.ends_with('.') {
                output.push('0');
            }
            output.push('_');
            output.push_str(suffix);
        }

        output
    }

    pub fn group_digits(output: &mut String, input: &str, group_size: usize, partial_group_first: bool, pad: bool) {
        debug_assert!(group_size > 0);

        let mut digits = input.chars().filter(|&c| c != '_');

        // The exponent may have a sign, output it early, otherwise it will be
        // treated as a digit
        if digits.clone().next() == Some('-') {
            let _: Option<char> = digits.next();
            output.push('-');
        }

        let first_group_size;

        if partial_group_first {
            first_group_size = (digits.clone().count() - 1) % group_size + 1;
            if pad {
                for _ in 0..group_size - first_group_size {
                    output.push('0');
                }
            }
        } else {
            first_group_size = group_size;
        }

        for _ in 0..first_group_size {
            if let Some(digit) = digits.next() {
                output.push(digit);
            }
        }

        for (c, i) in iter::zip(digits, (0..group_size).cycle()) {
            if i == 0 {
                output.push('_');
            }
            output.push(c);
        }
    }
}

fn split_suffix<'a>(src: &'a str, lit_kind: &LitKind) -> (&'a str, Option<&'a str>) {
    debug_assert!(lit_kind.is_numeric());
    lit_suffix_length(lit_kind)
        .and_then(|suffix_length| src.len().checked_sub(suffix_length))
        .map_or((src, None), |split_pos| {
            let (unsuffixed, suffix) = src.split_at(split_pos);
            (unsuffixed, Some(suffix))
        })
}

fn lit_suffix_length(lit_kind: &LitKind) -> Option<usize> {
    debug_assert!(lit_kind.is_numeric());
    let suffix = match lit_kind {
        LitKind::Int(_, int_lit_kind) => match int_lit_kind {
            LitIntType::Signed(int_ty) => Some(int_ty.name_str()),
            LitIntType::Unsigned(uint_ty) => Some(uint_ty.name_str()),
            LitIntType::Unsuffixed => None,
        },
        LitKind::Float(_, float_lit_kind) => match float_lit_kind {
            LitFloatType::Suffixed(float_ty) => Some(float_ty.name_str()),
            LitFloatType::Unsuffixed => None,
        },
        _ => None,
    };

    suffix.map(str::len)
}
