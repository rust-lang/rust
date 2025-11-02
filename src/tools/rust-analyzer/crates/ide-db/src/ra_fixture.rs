//! Working with the fixtures in r-a tests, and providing IDE services for them.

use std::hash::{BuildHasher, Hash};

use hir::{CfgExpr, FilePositionWrapper, FileRangeWrapper, Semantics};
use smallvec::SmallVec;
use span::{TextRange, TextSize};
use syntax::{
    AstToken, SmolStr,
    ast::{self, IsString},
};

use crate::{
    MiniCore, RootDatabase, SymbolKind, active_parameter::ActiveParameter,
    documentation::Documentation, range_mapper::RangeMapper, search::ReferenceCategory,
};

pub use span::FileId;

impl RootDatabase {
    fn from_ra_fixture(
        text: &str,
        minicore: MiniCore<'_>,
    ) -> Result<(RootDatabase, Vec<(FileId, usize)>, Vec<FileId>), ()> {
        // We don't want a mistake in the fixture to crash r-a, so we wrap this in `catch_unwind()`.
        std::panic::catch_unwind(|| {
            let mut db = RootDatabase::default();
            let fixture = test_fixture::ChangeFixture::parse_with_proc_macros(
                &db,
                text,
                minicore.0,
                Vec::new(),
            );
            db.apply_change(fixture.change);
            let files = fixture
                .files
                .into_iter()
                .zip(fixture.file_lines)
                .map(|(file_id, range)| (file_id.file_id(&db), range))
                .collect();
            (db, files, fixture.sysroot_files)
        })
        .map_err(|error| {
            tracing::error!(
                "cannot crate the crate graph: {}\nCrate graph:\n{}\n",
                if let Some(&s) = error.downcast_ref::<&'static str>() {
                    s
                } else if let Some(s) = error.downcast_ref::<String>() {
                    s.as_str()
                } else {
                    "Box<dyn Any>"
                },
                text,
            );
        })
    }
}

pub struct RaFixtureAnalysis {
    pub db: RootDatabase,
    tmp_file_ids: Vec<(FileId, usize)>,
    line_offsets: Vec<TextSize>,
    virtual_file_id_to_line: Vec<usize>,
    mapper: RangeMapper,
    literal: ast::String,
    // `minicore` etc..
    sysroot_files: Vec<FileId>,
    combined_len: TextSize,
}

impl RaFixtureAnalysis {
    pub fn analyze_ra_fixture(
        sema: &Semantics<'_, RootDatabase>,
        literal: ast::String,
        expanded: &ast::String,
        minicore: MiniCore<'_>,
        on_cursor: &mut dyn FnMut(TextRange),
    ) -> Option<RaFixtureAnalysis> {
        if !literal.is_raw() {
            return None;
        }

        let active_parameter = ActiveParameter::at_token(sema, expanded.syntax().clone())?;
        let has_rust_fixture_attr = active_parameter.attrs().is_some_and(|attrs| {
            attrs.filter_map(|attr| attr.as_simple_path()).any(|path| {
                path.segments()
                    .zip(["rust_analyzer", "rust_fixture"])
                    .all(|(seg, name)| seg.name_ref().map_or(false, |nr| nr.text() == name))
            })
        });
        if !has_rust_fixture_attr {
            return None;
        }
        let value = literal.value().ok()?;

        let mut mapper = RangeMapper::default();

        // This is used for the `Injector`, to resolve precise location in the string literal,
        // which will then be used to resolve precise location in the enclosing file.
        let mut offset_with_indent = TextSize::new(0);
        // This is used to resolve the location relative to the virtual file into a location
        // relative to the indentation-trimmed file which will then (by the `Injector`) used
        // to resolve to a location in the actual file.
        // Besides indentation, we also skip `$0` cursors for this, since they are not included
        // in the virtual files.
        let mut offset_without_indent = TextSize::new(0);

        let mut text = &*value;
        if let Some(t) = text.strip_prefix('\n') {
            offset_with_indent += TextSize::of("\n");
            text = t;
        }
        // This stores the offsets of each line, **after we remove indentation**.
        let mut line_offsets = Vec::new();
        for mut line in text.split_inclusive('\n') {
            line_offsets.push(offset_without_indent);

            if line.starts_with("@@") {
                // Introducing `//` into a fixture inside fixture causes all sorts of problems,
                // so for testing purposes we escape it as `@@` and replace it here.
                mapper.add("//", TextRange::at(offset_with_indent, TextSize::of("@@")));
                line = &line["@@".len()..];
                offset_with_indent += TextSize::of("@@");
                offset_without_indent += TextSize::of("@@");
            }

            // Remove indentation to simplify the mapping with fixture (which de-indents).
            // Removing indentation shouldn't affect highlighting.
            let mut unindented_line = line.trim_start();
            if unindented_line.is_empty() {
                // The whole line was whitespaces, but we need the newline.
                unindented_line = "\n";
            }
            offset_with_indent += TextSize::of(line) - TextSize::of(unindented_line);

            let marker = "$0";
            match unindented_line.find(marker) {
                Some(marker_pos) => {
                    let (before_marker, after_marker) = unindented_line.split_at(marker_pos);
                    let after_marker = &after_marker[marker.len()..];

                    mapper.add(
                        before_marker,
                        TextRange::at(offset_with_indent, TextSize::of(before_marker)),
                    );
                    offset_with_indent += TextSize::of(before_marker);
                    offset_without_indent += TextSize::of(before_marker);

                    if let Some(marker_range) = literal
                        .map_range_up(TextRange::at(offset_with_indent, TextSize::of(marker)))
                    {
                        on_cursor(marker_range);
                    }
                    offset_with_indent += TextSize::of(marker);

                    mapper.add(
                        after_marker,
                        TextRange::at(offset_with_indent, TextSize::of(after_marker)),
                    );
                    offset_with_indent += TextSize::of(after_marker);
                    offset_without_indent += TextSize::of(after_marker);
                }
                None => {
                    mapper.add(
                        unindented_line,
                        TextRange::at(offset_with_indent, TextSize::of(unindented_line)),
                    );
                    offset_with_indent += TextSize::of(unindented_line);
                    offset_without_indent += TextSize::of(unindented_line);
                }
            }
        }

        let combined = mapper.take_text();
        let combined_len = TextSize::of(&combined);
        let (analysis, tmp_file_ids, sysroot_files) =
            RootDatabase::from_ra_fixture(&combined, minicore).ok()?;

        // We use a `Vec` because we know the `FileId`s will always be close.
        let mut virtual_file_id_to_line = Vec::new();
        for &(file_id, line) in &tmp_file_ids {
            virtual_file_id_to_line.resize(file_id.index() as usize + 1, usize::MAX);
            virtual_file_id_to_line[file_id.index() as usize] = line;
        }

        Some(RaFixtureAnalysis {
            db: analysis,
            tmp_file_ids,
            line_offsets,
            virtual_file_id_to_line,
            mapper,
            literal,
            sysroot_files,
            combined_len,
        })
    }

    pub fn files(&self) -> impl Iterator<Item = FileId> {
        self.tmp_file_ids.iter().map(|(file, _)| *file)
    }

    /// This returns `None` for minicore or other sysroot files.
    fn virtual_file_id_to_line(&self, file_id: FileId) -> Option<usize> {
        if self.is_sysroot_file(file_id) {
            None
        } else {
            Some(self.virtual_file_id_to_line[file_id.index() as usize])
        }
    }

    pub fn map_offset_down(&self, offset: TextSize) -> Option<(FileId, TextSize)> {
        let inside_literal_range = self.literal.map_offset_down(offset)?;
        let combined_offset = self.mapper.map_offset_down(inside_literal_range)?;
        // There is usually a small number of files, so a linear search is smaller and faster.
        let (_, &(file_id, file_line)) =
            self.tmp_file_ids.iter().enumerate().find(|&(idx, &(_, file_line))| {
                let file_start = self.line_offsets[file_line];
                let file_end = self
                    .tmp_file_ids
                    .get(idx + 1)
                    .map(|&(_, next_file_line)| self.line_offsets[next_file_line])
                    .unwrap_or_else(|| self.combined_len);
                TextRange::new(file_start, file_end).contains(combined_offset)
            })?;
        let file_line_offset = self.line_offsets[file_line];
        let file_offset = combined_offset - file_line_offset;
        Some((file_id, file_offset))
    }

    pub fn map_range_down(&self, range: TextRange) -> Option<(FileId, TextRange)> {
        let (start_file_id, start_offset) = self.map_offset_down(range.start())?;
        let (end_file_id, end_offset) = self.map_offset_down(range.end())?;
        if start_file_id != end_file_id {
            None
        } else {
            Some((start_file_id, TextRange::new(start_offset, end_offset)))
        }
    }

    pub fn map_range_up(
        &self,
        virtual_file: FileId,
        range: TextRange,
    ) -> impl Iterator<Item = TextRange> {
        // This could be `None` if the file is empty.
        self.virtual_file_id_to_line(virtual_file)
            .and_then(|line| self.line_offsets.get(line))
            .into_iter()
            .flat_map(move |&tmp_file_offset| {
                // Resolve the offset relative to the virtual file to an offset relative to the combined indentation-trimmed file
                let range = range + tmp_file_offset;
                // Then resolve that to an offset relative to the real file.
                self.mapper.map_range_up(range)
            })
            // And finally resolve the offset relative to the literal to relative to the file.
            .filter_map(|range| self.literal.map_range_up(range))
    }

    pub fn map_offset_up(&self, virtual_file: FileId, offset: TextSize) -> Option<TextSize> {
        self.map_range_up(virtual_file, TextRange::empty(offset)).next().map(|range| range.start())
    }

    pub fn is_sysroot_file(&self, file_id: FileId) -> bool {
        self.sysroot_files.contains(&file_id)
    }
}

pub trait UpmapFromRaFixture: Sized {
    fn upmap_from_ra_fixture(
        self,
        analysis: &RaFixtureAnalysis,
        virtual_file_id: FileId,
        real_file_id: FileId,
    ) -> Result<Self, ()>;
}

trait IsEmpty {
    fn is_empty(&self) -> bool;
}

impl<T> IsEmpty for Vec<T> {
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

impl<T, const N: usize> IsEmpty for SmallVec<[T; N]> {
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

#[allow(clippy::disallowed_types)]
impl<K, V, S> IsEmpty for std::collections::HashMap<K, V, S> {
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

fn upmap_collection<T, Collection>(
    collection: Collection,
    analysis: &RaFixtureAnalysis,
    virtual_file_id: FileId,
    real_file_id: FileId,
) -> Result<Collection, ()>
where
    T: UpmapFromRaFixture,
    Collection: IntoIterator<Item = T> + FromIterator<T> + IsEmpty,
{
    if collection.is_empty() {
        // The collection was already empty, don't mark it as failing just because of that.
        return Ok(collection);
    }
    let result = collection
        .into_iter()
        .filter_map(|item| item.upmap_from_ra_fixture(analysis, virtual_file_id, real_file_id).ok())
        .collect::<Collection>();
    if result.is_empty() {
        // The collection was emptied by the upmapping - all items errored, therefore mark it as erroring as well.
        Err(())
    } else {
        Ok(result)
    }
}

impl<T: UpmapFromRaFixture> UpmapFromRaFixture for Option<T> {
    fn upmap_from_ra_fixture(
        self,
        analysis: &RaFixtureAnalysis,
        virtual_file_id: FileId,
        real_file_id: FileId,
    ) -> Result<Self, ()> {
        Ok(match self {
            Some(it) => Some(it.upmap_from_ra_fixture(analysis, virtual_file_id, real_file_id)?),
            None => None,
        })
    }
}

impl<T: UpmapFromRaFixture> UpmapFromRaFixture for Vec<T> {
    fn upmap_from_ra_fixture(
        self,
        analysis: &RaFixtureAnalysis,
        virtual_file_id: FileId,
        real_file_id: FileId,
    ) -> Result<Self, ()> {
        upmap_collection(self, analysis, virtual_file_id, real_file_id)
    }
}

impl<T: UpmapFromRaFixture, const N: usize> UpmapFromRaFixture for SmallVec<[T; N]> {
    fn upmap_from_ra_fixture(
        self,
        analysis: &RaFixtureAnalysis,
        virtual_file_id: FileId,
        real_file_id: FileId,
    ) -> Result<Self, ()> {
        upmap_collection(self, analysis, virtual_file_id, real_file_id)
    }
}

#[allow(clippy::disallowed_types)]
impl<K: UpmapFromRaFixture + Hash + Eq, V: UpmapFromRaFixture, S: BuildHasher + Default>
    UpmapFromRaFixture for std::collections::HashMap<K, V, S>
{
    fn upmap_from_ra_fixture(
        self,
        analysis: &RaFixtureAnalysis,
        virtual_file_id: FileId,
        real_file_id: FileId,
    ) -> Result<Self, ()> {
        upmap_collection(self, analysis, virtual_file_id, real_file_id)
    }
}

// A map of `FileId`s is treated as associating the ranges in the values with the keys.
#[allow(clippy::disallowed_types)]
impl<V: UpmapFromRaFixture, S: BuildHasher + Default> UpmapFromRaFixture
    for std::collections::HashMap<FileId, V, S>
{
    fn upmap_from_ra_fixture(
        self,
        analysis: &RaFixtureAnalysis,
        _virtual_file_id: FileId,
        real_file_id: FileId,
    ) -> Result<Self, ()> {
        if self.is_empty() {
            return Ok(self);
        }
        let result = self
            .into_iter()
            .filter_map(|(virtual_file_id, value)| {
                Some((
                    real_file_id,
                    value.upmap_from_ra_fixture(analysis, virtual_file_id, real_file_id).ok()?,
                ))
            })
            .collect::<std::collections::HashMap<_, _, _>>();
        if result.is_empty() { Err(()) } else { Ok(result) }
    }
}

macro_rules! impl_tuple {
    () => {}; // Base case.
    ( $first:ident, $( $rest:ident, )* ) => {
        impl<
            $first: UpmapFromRaFixture,
            $( $rest: UpmapFromRaFixture, )*
        > UpmapFromRaFixture for ( $first, $( $rest, )* ) {
            fn upmap_from_ra_fixture(
                self,
                analysis: &RaFixtureAnalysis,
                virtual_file_id: FileId,
                real_file_id: FileId,
            ) -> Result<Self, ()> {
                #[allow(non_snake_case)]
                let ( $first, $($rest,)* ) = self;
                Ok((
                    $first.upmap_from_ra_fixture(analysis, virtual_file_id, real_file_id)?,
                    $( $rest.upmap_from_ra_fixture(analysis, virtual_file_id, real_file_id)?, )*
                ))
            }
        }

        impl_tuple!( $($rest,)* );
    };
}
impl_tuple!(A, B, C, D, E,);

impl UpmapFromRaFixture for TextSize {
    fn upmap_from_ra_fixture(
        self,
        analysis: &RaFixtureAnalysis,
        virtual_file_id: FileId,
        _real_file_id: FileId,
    ) -> Result<Self, ()> {
        analysis.map_offset_up(virtual_file_id, self).ok_or(())
    }
}

impl UpmapFromRaFixture for TextRange {
    fn upmap_from_ra_fixture(
        self,
        analysis: &RaFixtureAnalysis,
        virtual_file_id: FileId,
        _real_file_id: FileId,
    ) -> Result<Self, ()> {
        analysis.map_range_up(virtual_file_id, self).next().ok_or(())
    }
}

// Deliberately do not implement that, as it's easy to get things misbehave and be treated with the wrong FileId:
//
// impl UpmapFromRaFixture for FileId {
//     fn upmap_from_ra_fixture(
//         self,
//         _analysis: &RaFixtureAnalysis,
//         _virtual_file_id: FileId,
//         real_file_id: FileId,
//     ) -> Result<Self, ()> {
//         Ok(real_file_id)
//     }
// }

impl UpmapFromRaFixture for FilePositionWrapper<FileId> {
    fn upmap_from_ra_fixture(
        self,
        analysis: &RaFixtureAnalysis,
        _virtual_file_id: FileId,
        real_file_id: FileId,
    ) -> Result<Self, ()> {
        Ok(FilePositionWrapper {
            file_id: real_file_id,
            offset: self.offset.upmap_from_ra_fixture(analysis, self.file_id, real_file_id)?,
        })
    }
}

impl UpmapFromRaFixture for FileRangeWrapper<FileId> {
    fn upmap_from_ra_fixture(
        self,
        analysis: &RaFixtureAnalysis,
        _virtual_file_id: FileId,
        real_file_id: FileId,
    ) -> Result<Self, ()> {
        Ok(FileRangeWrapper {
            file_id: real_file_id,
            range: self.range.upmap_from_ra_fixture(analysis, self.file_id, real_file_id)?,
        })
    }
}

#[macro_export]
macro_rules! impl_empty_upmap_from_ra_fixture {
    ( $( $ty:ty ),* $(,)? ) => {
        $(
            impl $crate::ra_fixture::UpmapFromRaFixture for $ty {
                fn upmap_from_ra_fixture(
                    self,
                    _analysis: &$crate::ra_fixture::RaFixtureAnalysis,
                    _virtual_file_id: $crate::ra_fixture::FileId,
                    _real_file_id: $crate::ra_fixture::FileId,
                ) -> Result<Self, ()> {
                    Ok(self)
                }
            }
        )*
    };
}

impl_empty_upmap_from_ra_fixture!(
    bool,
    i8,
    i16,
    i32,
    i64,
    i128,
    u8,
    u16,
    u32,
    u64,
    u128,
    f32,
    f64,
    &str,
    String,
    SmolStr,
    Documentation,
    SymbolKind,
    CfgExpr,
    ReferenceCategory,
);
