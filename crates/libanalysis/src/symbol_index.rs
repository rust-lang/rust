use libeditor::{FileSymbol, file_symbols};
use libsyntax2::{
    File,
    SyntaxKind::{self, *},
};
use fst::{self, Streamer};
use {Query, FileId, JobToken};

#[derive(Debug)]
pub(crate) struct FileSymbols {
    symbols: Vec<FileSymbol>,
    map: fst::Map,
}

impl FileSymbols {
    pub(crate) fn new(file: &File) -> FileSymbols {
        let mut symbols = file_symbols(file)
            .into_iter()
            .map(|s| (s.name.as_str().to_lowercase(), s))
            .collect::<Vec<_>>();

        symbols.sort_by(|s1, s2| s1.0.cmp(&s2.0));
        symbols.dedup_by(|s1, s2| s1.0 == s2.0);
        let (names, symbols): (Vec<String>, Vec<FileSymbol>) =
            symbols.into_iter().unzip();

        let map = fst::Map::from_iter(
            names.into_iter().zip(0u64..)
        ).unwrap();
        FileSymbols { symbols, map }
    }
}

impl Query {
    pub(crate) fn search(
        mut self,
        indices: &[(FileId, &FileSymbols)],
        token: &JobToken,
    ) -> Vec<(FileId, FileSymbol)> {

        let mut op = fst::map::OpBuilder::new();
        for (_, file_symbols) in indices.iter() {
            let automaton = fst::automaton::Subsequence::new(&self.lowercased);
            op = op.add(file_symbols.map.search(automaton))
        }
        let mut stream = op.union();
        let mut res = Vec::new();
        while let Some((_, indexed_values)) = stream.next() {
            if self.limit == 0 || token.is_canceled() {
                break;
            }
            for indexed_value in indexed_values {
                let (file_id, file_symbols) = &indices[indexed_value.index];
                let idx = indexed_value.value as usize;

                let symbol = &file_symbols.symbols[idx];
                if self.only_types && !is_type(symbol.kind) {
                    continue;
                }
                if self.exact && symbol.name != self.query {
                    continue;
                }
                res.push((*file_id, symbol.clone()));
                self.limit -= 1;
            }
        }
        res
    }
}

fn is_type(kind: SyntaxKind) -> bool {
    match kind {
        STRUCT_DEF | ENUM_DEF | TRAIT_DEF | TYPE_DEF => true,
        _ => false,
    }
}
