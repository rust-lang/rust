use libeditor::{FileSymbol, file_symbols};
use libsyntax2::{
    ast,
    SyntaxKind::{self, *},
};
use fst::{self, IntoStreamer};

use Search;

#[derive(Debug)]
pub(crate) struct FileSymbols {
    symbols: Vec<FileSymbol>,
    map: fst::Map,
}

impl FileSymbols {
    pub(crate) fn new(file: &ast::File) -> FileSymbols {
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

pub(crate) struct Query {
    query: String,
    all_symbols: bool,
}

impl Query {
    pub(crate) fn new(query: &str) -> Query {
        let all_symbols = query.contains("#");
        let query: String = query.chars()
            .filter(|&c| c != '#')
            .flat_map(char::to_lowercase)
            .collect();
        Query { query, all_symbols }
    }

    pub(crate) fn process(
        &self,
        file: &FileSymbols,
        acc: &mut FnMut(&FileSymbol) -> Search,
    ) -> Search {
        fn is_type(kind: SyntaxKind) -> bool {
            match kind {
                STRUCT | ENUM | TRAIT | TYPE_ITEM => true,
                _ => false,
            }
        }
        let automaton = fst::automaton::Subsequence::new(&self.query);
        for idx in file.map.search(automaton).into_stream().into_values() {
            let idx = idx as usize;
            let symbol = &file.symbols[idx];
            if self.all_symbols || is_type(symbol.kind) {
                if acc(&symbol) == Search::Break {
                    return Search::Break;
                }
            }
        }
        Search::Continue
    }
}

