use arrayvec::ArrayVec;
use core::str::FromStr as _;
use itertools::Itertools as _;
use rustc_attr_parsing::parse_version;
use rustc_data_structures::hash_map::HashMap;
use rustc_data_structures::hash_set::HashSet;
use rustc_errors::{DiagCtxtHandle, DiagMessage};
use rustc_hir::attrs::RustcVersion;
use rustc_session::Session;
use rustc_session::config::ErrorOutputType;
use rustc_span::edit_distance::edit_distance;
use rustc_span::{BytePos, Pos as _, Span, Spanned, Symbol};
use std::fmt::{self, Display, Write as _};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::ops::{ControlFlow, Range};
use toml::de::DeValue;

pub type TomlValue<'a> = toml::Spanned<DeValue<'a>>;

pub struct DiagCtxt<'a> {
    pub inner: DiagCtxtHandle<'a>,
    pub width: usize,
    pub render_pretty: bool,
    offset: usize,
}
impl<'a> DiagCtxt<'a> {
    pub fn new(sess: &'a Session, offset: usize) -> Self {
        Self {
            inner: sess.dcx(),
            width: sess.diagnostic_width(),
            render_pretty: match sess.opts.error_format {
                ErrorOutputType::HumanReadable { .. } => true,
                ErrorOutputType::Json { pretty, .. } => pretty,
            },
            offset,
        }
    }

    pub fn make_sp(&self, range: Range<usize>) -> Span {
        Span::with_root_ctxt(
            BytePos::from_usize(self.offset + range.start),
            BytePos::from_usize(self.offset + range.end),
        )
    }

    pub fn span_err(&self, range: Range<usize>, msg: impl Into<DiagMessage>) {
        self.inner.span_err(self.make_sp(range), msg);
    }

    pub fn span_warn(&self, range: Range<usize>, msg: impl Into<DiagMessage>) {
        self.inner.span_warn(self.make_sp(range), msg);
    }
}

/// Attempts to find the closest matching string from the list. Returns `None`
/// if the edit distance is too large.
pub fn find_closest_match<'a>(s: &str, options: &[&'a str]) -> Option<&'a str> {
    // Don't treat `_` to `-` and case conversions as an edit.
    let mut s = s.replace('_', "-");
    s.make_ascii_lowercase();
    options
        .iter()
        .filter_map(|&option| edit_distance(&s, &option.to_ascii_lowercase(), 4).map(|x| (x, option)))
        .min_by_key(|&(dist, _)| dist)
        .map(|(_, x)| x)
}

/// Creates a message listing all possible values suitable for use in `Diag::note`.
pub fn create_value_list_msg(dcx: &DiagCtxt<'_>, values: &[&str]) -> String {
    const NOTE_WITH_MSG_LEN: usize = "   = note: possible values: ".len();
    const NOTE_LEN: usize = "   = note: ".len();
    const TBL_SEP: &str = "    ";
    const INLINE_SEP: &str = ", ";

    // Print everything on one line if it will fit and there aren't too many values.
    // e.g. "note: possible values: `value1`, `value2`, `value3`"
    //
    // If there are too many values and the note would exceed the terminal width lay the
    // values out into columns. e.g.
    //    possible values:
    //    value1    value5
    //    value2    value6
    //    value3    value7
    //    value4
    if values.len() <= 8
        && NOTE_WITH_MSG_LEN + values.iter().map(|x| x.len() + INLINE_SEP.len() + 2).sum::<usize>() <= dcx.width
    {
        format!(
            "possible values: {}",
            values.iter().format_with(INLINE_SEP, |x, f| f(&format_args!("`{x}`"))),
        )
    } else if dcx.render_pretty {
        // The minimum width a column could possibly have.
        let min_width = values.iter().map(|x| x.len()).min().unwrap_or(0);
        // The number of columns that fit using the minimum width.
        let max_col = dcx.width.saturating_sub(NOTE_WITH_MSG_LEN) / (min_width + TBL_SEP.len());

        // Determine the starting dimensions of the search.
        let start_size = (2..=max_col).try_fold(values.len(), |row_count, col_count| {
            let needed_rows = values.len().div_ceil(col_count);
            // Only add a new column if it will remove several rows.
            if needed_rows + 3 <= row_count {
                ControlFlow::Continue(needed_rows)
            } else {
                ControlFlow::Break((row_count, col_count - 1))
            }
        });
        let (mut row_count, init_col_count) = match start_size {
            // Also handles the case where `max_col < 2`
            ControlFlow::Continue(x) => (x, max_col),
            ControlFlow::Break(x) => x,
        };

        // The current total width required.
        let mut total_width = init_col_count * (min_width + TBL_SEP.len()) + NOTE_LEN;
        // The current width of each column without the prefix.
        let mut col_widths = vec![min_width; init_col_count];

        // Determine the required width of each column.
        'outer: loop {
            // Also handles the case where `max_col` is zero.
            if col_widths.len() <= 1 {
                return format!(
                    "possible values:\n{}",
                    values.iter().format_with("\n", |x, f| f(&format_args!("{x}"))),
                );
            }
            for (col_values, col_width) in values.chunks(row_count).zip(col_widths.iter_mut()) {
                for value in col_values {
                    if value.len() > *col_width {
                        let delta = value.len() - *col_width;
                        *col_width += delta;
                        total_width += delta;
                        if total_width > dcx.width {
                            // Remove a column and reset the metrics then retry.
                            col_widths.pop();
                            row_count = values.len().div_ceil(col_widths.len());
                            col_widths.fill(min_width);
                            total_width = col_widths.len() * (min_width + TBL_SEP.len()) + NOTE_LEN;
                            continue 'outer;
                        }
                    }
                }
            }
            break;
        }

        format!(
            "possible values:\n{}",
            (0..row_count).format_with("\n", |row, f| {
                f(&(row..values.len())
                    .step_by(row_count)
                    .zip(&col_widths)
                    .format_with(TBL_SEP, |(i, &width), f| {
                        // Don't print trailing whitespace on the right edge.
                        let width = if i + row_count < values.len() { width } else { 0 };
                        f(&format_args!("{:width$}", values[i]))
                    }))
            }),
        )
    } else {
        format!(
            "possible values:{}",
            values.iter().format_with("", |x, f| f(&format_args!("\n{x}"))),
        )
    }
}

/// A type which can be constructed from a default value.
pub trait FromDefault<T>: Sized {
    /// Creates this value from a default value.
    fn from_default(default: T) -> Self;
    /// Writes the default value to a string.
    fn display_default(default: T) -> impl Display;
}

macro_rules! impl_from_default_passthrough {
    ($($ty:ty),*) => {$(
        impl FromDefault<$ty> for $ty {
            #[inline]
            fn from_default(default: $ty) -> Self {
                default
            }
            #[inline]
            fn display_default(default: $ty) -> impl Display {
                default
            }
        }
    )*}
}
impl_from_default_passthrough!(bool, u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

impl<T> FromDefault<()> for Option<T> {
    fn from_default((): ()) -> Self {
        None
    }
    fn display_default((): ()) -> impl Display {
        // will cause an error in the metadata collector
        ""
    }
}

struct DisplayStr(&'static str);
impl Display for DisplayStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.0, f)
    }
}

impl FromDefault<&'static str> for String {
    fn from_default(default: &'static str) -> Self {
        default.into()
    }
    fn display_default(default: &'static str) -> impl Display {
        DisplayStr(default)
    }
}

impl<T> FromDefault<()> for Vec<T> {
    fn from_default((): ()) -> Self {
        Vec::new()
    }
    fn display_default((): ()) -> impl Display {
        "[]"
    }
}

impl<T, S: Default> FromDefault<()> for HashSet<T, S> {
    fn from_default((): ()) -> Self {
        HashSet::default()
    }
    fn display_default((): ()) -> impl Display {
        "[]"
    }
}

struct DisplaySlice<T: 'static, U>(&'static [T], PhantomData<U>);
impl<T, U> Display for DisplaySlice<T, U>
where
    T: 'static + Copy,
    U: FromDefault<T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char('[')?;
        if let Some((head, tail)) = self.0.split_first() {
            U::display_default(*head).fmt(f)?;
            for x in tail {
                write!(f, ", {}", U::display_default(*x))?;
            }
        }
        f.write_char(']')
    }
}

impl<T, U> FromDefault<&'static [U]> for Vec<T>
where
    T: FromDefault<U>,
    U: 'static + Copy,
{
    fn from_default(default: &'static [U]) -> Self {
        default.iter().map(|&x| T::from_default(x)).collect()
    }
    fn display_default(default: &'static [U]) -> impl Display {
        DisplaySlice::<_, T>(default, PhantomData)
    }
}

impl<T, U, S> FromDefault<&'static [U]> for HashSet<T, S>
where
    T: FromDefault<U> + Eq + Hash,
    U: 'static + Copy,
    S: Default + BuildHasher,
{
    fn from_default(default: &'static [U]) -> Self {
        default.iter().map(|&x| T::from_default(x)).collect()
    }
    fn display_default(default: &'static [U]) -> impl Display {
        DisplaySlice::<_, T>(default, PhantomData)
    }
}

/// A type which can be deserialized from a toml value.
pub trait Deserialize: Sized {
    /// Attempt to deserialize the value. Returns `None` and raises an error on failure.
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self>;
}

impl Deserialize for bool {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        match value.get_ref().as_bool() {
            None => {
                dcx.span_err(value.span(), "expected a boolean");
                None
            },
            x => x,
        }
    }
}

macro_rules! impl_deserialize_int {
    ($($ty:ident),*) => {$(
        impl Deserialize for $ty {
            fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
                match value.get_ref().as_integer() {
                    None => {
                        dcx.span_err(value.span(), "expected an integer");
                        None
                    },
                    Some(x) => match $ty::from_str_radix(x.as_str(), x.radix()) {
                        Ok(x) => Some(x),
                        Err(_) => {
                            dcx.span_err(
                                value.span(),
                                format!("integer is not within the expected range ({}..{})", $ty::MIN, $ty::MAX),
                            );
                            None
                        }
                    }
                }
            }
        }
    )*}
}
impl_deserialize_int!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

macro_rules! impl_deserialize_float {
    ($($ty:ident),*) => {$(
        impl Deserialize for $ty {
            fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
                match value.get_ref().as_float() {
                    None => {
                        dcx.span_err(value.span(), "expected a number");
                        None
                    },
                    Some(x) => match $ty::from_str(x.as_str()) {
                        Ok(x) => Some(x),
                        Err(_) => {
                            dcx.span_err(value.span(), "failed to parse number");
                            None
                        }
                    }
                }
            }
        }
    )*}
}
impl_deserialize_float!(f32, f64);

impl Deserialize for String {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        if let Some(x) = value.get_ref().as_str() {
            Some(x.into())
        } else {
            dcx.span_err(value.span(), "expected a string");
            None
        }
    }
}

impl Deserialize for Symbol {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        if let Some(x) = value.get_ref().as_str() {
            Some(Symbol::intern(x))
        } else {
            dcx.span_err(value.span(), "expected a string");
            None
        }
    }
}

impl Deserialize for RustcVersion {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        if let Some(x) = value.get_ref().as_str() {
            if let Some(x) = parse_version(Symbol::intern(x)) {
                Some(x)
            } else {
                dcx.span_err(value.span(), "failed to parse rust version");
                None
            }
        } else {
            dcx.span_err(value.span(), "expected a version string");
            None
        }
    }
}

impl<T: Deserialize, const N: usize> Deserialize for [T; N] {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        if let Some(values) = value.get_ref().as_array()
            && values.len() == N
        {
            let values = values
                .iter()
                .filter_map(|x| T::deserialize(dcx, x))
                .collect::<ArrayVec<T, N>>();
            // A value's deserialize impl will have already given an error
            values.into_inner().ok()
        } else {
            dcx.span_err(value.span(), "expected an array of length `{N}`");
            None
        }
    }
}

impl<T: Deserialize> Deserialize for Vec<T> {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        if let Some(values) = value.get_ref().as_array() {
            Some(values.iter().filter_map(|x| T::deserialize(dcx, x)).collect())
        } else {
            dcx.span_err(value.span(), "expected an array");
            None
        }
    }
}

impl<T, S> Deserialize for HashSet<T, S>
where
    T: Deserialize + Eq + Hash,
    S: Default + BuildHasher,
{
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        if let Some(values) = value.as_ref().as_array() {
            Some(values.iter().filter_map(|x| T::deserialize(dcx, x)).collect())
        } else {
            dcx.span_err(value.span(), "expected an array");
            None
        }
    }
}

impl<T, S> Deserialize for HashMap<T, Span, S>
where
    T: Deserialize + Eq + Hash,
    S: Default + BuildHasher,
{
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        if let Some(values) = value.as_ref().as_array() {
            Some(
                values
                    .iter()
                    .filter_map(|x| T::deserialize(dcx, x).map(|value| (value, dcx.make_sp(x.span()))))
                    .collect(),
            )
        } else {
            dcx.span_err(value.span(), "expected an array");
            None
        }
    }
}

impl<T: Deserialize> Deserialize for Spanned<T> {
    fn deserialize(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>) -> Option<Self> {
        T::deserialize(dcx, value).map(|x| Spanned {
            node: x,
            span: dcx.make_sp(value.span()),
        })
    }
}

/// A type which can be deserialized from a toml value with a fallback to a default value.
pub trait DeserializeOrDefault<T>: Sized {
    /// Attempt to deserialize the value. Returns the default value and raises an error on failure.
    fn deserialize_or_default(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>, default: T) -> Self;
}

impl<T: Deserialize + FromDefault<T>> DeserializeOrDefault<T> for T {
    fn deserialize_or_default(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>, default: T) -> Self {
        T::deserialize(dcx, value).unwrap_or_else(|| T::from_default(default))
    }
}

impl<T: Deserialize + Default> DeserializeOrDefault<()> for T {
    fn deserialize_or_default(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>, (): ()) -> Self {
        T::deserialize(dcx, value).unwrap_or_default()
    }
}

impl<T: Deserialize> DeserializeOrDefault<()> for Option<T> {
    fn deserialize_or_default(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>, (): ()) -> Self {
        T::deserialize(dcx, value)
    }
}

pub fn deserialize_array<T, U, C>(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>, default: &'static [U]) -> C
where
    T: Deserialize + FromDefault<U>,
    U: Copy,
    C: FromIterator<T> + Extend<T>,
{
    let default_iter = default.iter().map(|&x| T::from_default(x));
    if let Some(values) = value.as_ref().as_array() {
        let mut has_default = false;
        let mut res: C = values
            .iter()
            .filter(|x| {
                if x.as_ref().as_str().is_some_and(|x| x == "..") {
                    if has_default {
                        dcx.span_warn(value.span(), "duplicate `..` item");
                    }
                    has_default = true;
                    false
                } else {
                    true
                }
            })
            .filter_map(|x| T::deserialize(dcx, x))
            .collect();
        if has_default {
            res.extend(default_iter);
        }
        res
    } else {
        dcx.span_err(value.span(), "expected an array");
        default_iter.collect()
    }
}

impl<T, U> DeserializeOrDefault<&'static [U]> for Vec<T>
where
    T: Deserialize + FromDefault<U>,
    U: Copy,
{
    fn deserialize_or_default(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>, default: &'static [U]) -> Self {
        deserialize_array(dcx, value, default)
    }
}

impl<T, U, S> DeserializeOrDefault<&'static [U]> for HashSet<T, S>
where
    T: Deserialize + FromDefault<U> + Eq + Hash,
    U: Copy,
    S: Default + BuildHasher,
{
    fn deserialize_or_default(dcx: &DiagCtxt<'_>, value: &TomlValue<'_>, default: &'static [U]) -> Self {
        deserialize_array(dcx, value, default)
    }
}

macro_rules! deserialize_table {
    ($dcx:ident, $table:ident, $($name:ident($name_str:literal): $ty:ty,)+) => {
        $(let mut $name: Option<$ty> = None;)+

        for (key, value) in $table.iter() {
            match &**key.get_ref() {
                $($name_str => {
                    // Duplicate keys are handled by the toml parser
                    $name = <$ty as crate::de::Deserialize>::deserialize($dcx, value.into());
                },)+
                _ => {
                    const NAMES: &[&str] = &[$($name_str),*];
                    let sp = $dcx.make_sp(key.span());
                    let mut diag = $dcx.inner.struct_span_err(sp, "unknown key");
                    if let Some(sugg) = crate::de::find_closest_match(key.as_ref(), NAMES) {
                        diag.span_suggestion(sp, "did you mean", sugg, Applicability::MaybeIncorrect);
                    }
                    diag.note(crate::de::create_value_list_msg($dcx, NAMES));
                    diag.emit();
                },
            }
        }
    }
}
