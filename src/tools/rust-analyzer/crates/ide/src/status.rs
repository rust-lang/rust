use std::{fmt, marker::PhantomData};

use hir::{
    db::{AstIdMapQuery, AttrsQuery, BlockDefMapQuery, ParseMacroExpansionQuery},
    Attr, Attrs, ExpandResult, MacroFileId, Module,
};
use ide_db::{
    base_db::{
        ra_salsa::{
            debug::{DebugQueryTable, TableEntry},
            Query, QueryTable,
        },
        CompressedFileTextQuery, CrateData, ParseQuery, SourceDatabase, SourceRootId,
    },
    symbol_index::ModuleSymbolsQuery,
};
use ide_db::{
    symbol_index::{LibrarySymbolsQuery, SymbolIndex},
    RootDatabase,
};
use itertools::Itertools;
use profile::{memory_usage, Bytes};
use span::{EditionedFileId, FileId};
use stdx::format_to;
use syntax::{ast, Parse, SyntaxNode};
use triomphe::Arc;

// Feature: Status
//
// Shows internal statistic about memory usage of rust-analyzer.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **rust-analyzer: Status**
// |===
// image::https://user-images.githubusercontent.com/48062697/113065584-05f34500-91b1-11eb-98cc-5c196f76be7f.gif[]
pub(crate) fn status(db: &RootDatabase, file_id: Option<FileId>) -> String {
    let mut buf = String::new();

    format_to!(buf, "{}\n", collect_query(CompressedFileTextQuery.in_db(db)));
    format_to!(buf, "{}\n", collect_query(ParseQuery.in_db(db)));
    format_to!(buf, "{}\n", collect_query(ParseMacroExpansionQuery.in_db(db)));
    format_to!(buf, "{}\n", collect_query(LibrarySymbolsQuery.in_db(db)));
    format_to!(buf, "{}\n", collect_query(ModuleSymbolsQuery.in_db(db)));
    format_to!(buf, "{} in total\n", memory_usage());

    format_to!(buf, "\nDebug info:\n");
    format_to!(buf, "{}\n", collect_query(AttrsQuery.in_db(db)));
    format_to!(buf, "{} ast id maps\n", collect_query_count(AstIdMapQuery.in_db(db)));
    format_to!(buf, "{} block def maps\n", collect_query_count(BlockDefMapQuery.in_db(db)));

    if let Some(file_id) = file_id {
        format_to!(buf, "\nCrates for file {}:\n", file_id.index());
        let crates = crate::parent_module::crates_for(db, file_id);
        if crates.is_empty() {
            format_to!(buf, "Does not belong to any crate");
        }
        let crate_graph = db.crate_graph();
        for crate_id in crates {
            let CrateData {
                root_file_id,
                edition,
                version,
                display_name,
                cfg_options,
                potential_cfg_options,
                env,
                dependencies,
                origin,
                is_proc_macro,
            } = &crate_graph[crate_id];
            format_to!(
                buf,
                "Crate: {}\n",
                match display_name {
                    Some(it) => format!("{it}({})", crate_id.into_raw()),
                    None => format!("{}", crate_id.into_raw()),
                }
            );
            format_to!(buf, "    Root module file id: {}\n", root_file_id.index());
            format_to!(buf, "    Edition: {}\n", edition);
            format_to!(buf, "    Version: {}\n", version.as_deref().unwrap_or("n/a"));
            format_to!(buf, "    Enabled cfgs: {:?}\n", cfg_options);
            format_to!(buf, "    Potential cfgs: {:?}\n", potential_cfg_options);
            format_to!(buf, "    Env: {:?}\n", env);
            format_to!(buf, "    Origin: {:?}\n", origin);
            format_to!(buf, "    Is a proc macro crate: {}\n", is_proc_macro);
            let deps = dependencies
                .iter()
                .map(|dep| format!("{}={}", dep.name, dep.crate_id.into_raw()))
                .format(", ");
            format_to!(buf, "    Dependencies: {}\n", deps);
        }
    }

    buf.trim().to_owned()
}

fn collect_query<'q, Q>(table: QueryTable<'q, Q>) -> <Q as QueryCollect>::Collector
where
    QueryTable<'q, Q>: DebugQueryTable,
    Q: QueryCollect,
    <Q as Query>::Storage: 'q,
    <Q as QueryCollect>::Collector: StatCollect<
        <QueryTable<'q, Q> as DebugQueryTable>::Key,
        <QueryTable<'q, Q> as DebugQueryTable>::Value,
    >,
{
    struct StatCollectorWrapper<C>(C);
    impl<C: StatCollect<K, V>, K, V> FromIterator<TableEntry<K, V>> for StatCollectorWrapper<C> {
        fn from_iter<T>(iter: T) -> StatCollectorWrapper<C>
        where
            T: IntoIterator<Item = TableEntry<K, V>>,
        {
            let mut res = C::default();
            for entry in iter {
                res.collect_entry(entry.key, entry.value);
            }
            StatCollectorWrapper(res)
        }
    }
    table.entries::<StatCollectorWrapper<<Q as QueryCollect>::Collector>>().0
}

fn collect_query_count<'q, Q>(table: QueryTable<'q, Q>) -> usize
where
    QueryTable<'q, Q>: DebugQueryTable,
    Q: Query,
    <Q as Query>::Storage: 'q,
{
    struct EntryCounter(usize);
    impl<K, V> FromIterator<TableEntry<K, V>> for EntryCounter {
        fn from_iter<T>(iter: T) -> EntryCounter
        where
            T: IntoIterator<Item = TableEntry<K, V>>,
        {
            EntryCounter(iter.into_iter().count())
        }
    }
    table.entries::<EntryCounter>().0
}

trait QueryCollect: Query {
    type Collector;
}

impl QueryCollect for LibrarySymbolsQuery {
    type Collector = SymbolsStats<SourceRootId>;
}

impl QueryCollect for ParseQuery {
    type Collector = SyntaxTreeStats<false>;
}

impl QueryCollect for ParseMacroExpansionQuery {
    type Collector = SyntaxTreeStats<true>;
}

impl QueryCollect for CompressedFileTextQuery {
    type Collector = FilesStats;
}

impl QueryCollect for ModuleSymbolsQuery {
    type Collector = SymbolsStats<Module>;
}

impl QueryCollect for AttrsQuery {
    type Collector = AttrsStats;
}

trait StatCollect<K, V>: Default {
    fn collect_entry(&mut self, key: K, value: Option<V>);
}

#[derive(Default)]
struct FilesStats {
    total: usize,
    size: Bytes,
}

impl fmt::Display for FilesStats {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{} of files", self.size)
    }
}

impl StatCollect<FileId, Arc<[u8]>> for FilesStats {
    fn collect_entry(&mut self, _: FileId, value: Option<Arc<[u8]>>) {
        self.total += 1;
        self.size += value.unwrap().len();
    }
}

#[derive(Default)]
pub(crate) struct SyntaxTreeStats<const MACROS: bool> {
    total: usize,
    pub(crate) retained: usize,
}

impl<const MACROS: bool> fmt::Display for SyntaxTreeStats<MACROS> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            fmt,
            "{} trees, {} preserved{}",
            self.total,
            self.retained,
            if MACROS { " (macros)" } else { "" }
        )
    }
}

impl StatCollect<EditionedFileId, Parse<ast::SourceFile>> for SyntaxTreeStats<false> {
    fn collect_entry(&mut self, _: EditionedFileId, value: Option<Parse<ast::SourceFile>>) {
        self.total += 1;
        self.retained += value.is_some() as usize;
    }
}

impl<M> StatCollect<MacroFileId, ExpandResult<(Parse<SyntaxNode>, M)>> for SyntaxTreeStats<true> {
    fn collect_entry(
        &mut self,
        _: MacroFileId,
        value: Option<ExpandResult<(Parse<SyntaxNode>, M)>>,
    ) {
        self.total += 1;
        self.retained += value.is_some() as usize;
    }
}

struct SymbolsStats<Key> {
    total: usize,
    size: Bytes,
    phantom: PhantomData<Key>,
}

impl<Key> Default for SymbolsStats<Key> {
    fn default() -> Self {
        Self { total: Default::default(), size: Default::default(), phantom: PhantomData }
    }
}

impl fmt::Display for SymbolsStats<Module> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{} of module index symbols ({})", self.size, self.total)
    }
}
impl fmt::Display for SymbolsStats<SourceRootId> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{} of library index symbols ({})", self.size, self.total)
    }
}
impl<Key> StatCollect<Key, Arc<SymbolIndex>> for SymbolsStats<Key> {
    fn collect_entry(&mut self, _: Key, value: Option<Arc<SymbolIndex>>) {
        if let Some(symbols) = value {
            self.total += symbols.len();
            self.size += symbols.memory_size();
        }
    }
}

#[derive(Default)]
struct AttrsStats {
    entries: usize,
    total: usize,
}

impl fmt::Display for AttrsStats {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let size =
            self.entries * std::mem::size_of::<Attrs>() + self.total * std::mem::size_of::<Attr>();
        let size = Bytes::new(size as _);
        write!(
            fmt,
            "{} attribute query entries, {} total attributes ({} for storing entries)",
            self.entries, self.total, size
        )
    }
}

impl<Key> StatCollect<Key, Attrs> for AttrsStats {
    fn collect_entry(&mut self, _: Key, value: Option<Attrs>) {
        self.entries += 1;
        self.total += value.map_or(0, |it| it.len());
    }
}
