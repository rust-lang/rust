//! Benchmark operations like highlighting or goto definition.

use std::{env, path::PathBuf, str::FromStr, sync::Arc, time::Instant};

use anyhow::{bail, format_err, Result};
use hir::PrefixKind;
use ide::{
    Analysis, AnalysisHost, Change, CompletionConfig, DiagnosticsConfig, FilePosition, LineCol,
};
use ide_db::{
    base_db::{
        salsa::{Database, Durability},
        FileId,
    },
    helpers::{insert_use::InsertUseConfig, SnippetCap},
};
use vfs::AbsPathBuf;

use crate::cli::{load_cargo::load_cargo, print_memory_usage, Verbosity};

pub struct BenchCmd {
    pub path: PathBuf,
    pub what: BenchWhat,
    pub memory_usage: bool,
    pub load_output_dirs: bool,
    pub with_proc_macro: bool,
}

pub enum BenchWhat {
    Highlight { path: AbsPathBuf },
    Complete(Position),
    GotoDef(Position),
}

pub struct Position {
    pub path: AbsPathBuf,
    pub line: u32,
    pub column: u32,
}

impl FromStr for Position {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let mut split = s.rsplitn(3, ':');
        match (split.next(), split.next(), split.next()) {
            (Some(column), Some(line), Some(path)) => {
                let path = env::current_dir().unwrap().join(path);
                let path = AbsPathBuf::assert(path);
                Ok(Position { path, line: line.parse()?, column: column.parse()? })
            }
            _ => bail!("position should be in file:line:column format: {:?}", s),
        }
    }
}

impl BenchCmd {
    pub fn run(self, verbosity: Verbosity) -> Result<()> {
        profile::init();

        let start = Instant::now();
        eprint!("loading: ");
        let (mut host, vfs) = load_cargo(&self.path, self.load_output_dirs, self.with_proc_macro)?;
        eprintln!("{:?}\n", start.elapsed());

        let file_id = {
            let path = match &self.what {
                BenchWhat::Highlight { path } => path,
                BenchWhat::Complete(pos) | BenchWhat::GotoDef(pos) => &pos.path,
            };
            let path = path.clone().into();
            vfs.file_id(&path).ok_or_else(|| format_err!("Can't find {}", path))?
        };

        match &self.what {
            BenchWhat::Highlight { .. } => {
                let res = do_work(&mut host, file_id, |analysis| {
                    analysis.diagnostics(&DiagnosticsConfig::default(), file_id).unwrap();
                    analysis.highlight_as_html(file_id, false).unwrap()
                });
                if verbosity.is_verbose() {
                    println!("\n{}", res);
                }
            }
            BenchWhat::Complete(pos) | BenchWhat::GotoDef(pos) => {
                let is_completion = matches!(self.what, BenchWhat::Complete(..));

                let offset = host
                    .analysis()
                    .file_line_index(file_id)?
                    .offset(LineCol { line: pos.line - 1, col_utf16: pos.column });
                let file_position = FilePosition { file_id, offset };

                if is_completion {
                    let options = CompletionConfig {
                        enable_postfix_completions: true,
                        enable_imports_on_the_fly: true,
                        add_call_parenthesis: true,
                        add_call_argument_snippets: true,
                        snippet_cap: SnippetCap::new(true),
                        insert_use: InsertUseConfig { merge: None, prefix_kind: PrefixKind::Plain },
                    };
                    let res = do_work(&mut host, file_id, |analysis| {
                        analysis.completions(&options, file_position)
                    });
                    if verbosity.is_verbose() {
                        println!("\n{:#?}", res);
                    }
                } else {
                    let res = do_work(&mut host, file_id, |analysis| {
                        analysis.goto_definition(file_position)
                    });
                    if verbosity.is_verbose() {
                        println!("\n{:#?}", res);
                    }
                }
            }
        }

        if self.memory_usage {
            print_memory_usage(host, vfs);
        }

        Ok(())
    }
}

fn do_work<F: Fn(&Analysis) -> T, T>(host: &mut AnalysisHost, file_id: FileId, work: F) -> T {
    {
        let start = Instant::now();
        eprint!("from scratch:   ");
        work(&host.analysis());
        eprintln!("{:?}", start.elapsed());
    }
    {
        let start = Instant::now();
        eprint!("no change:      ");
        work(&host.analysis());
        eprintln!("{:?}", start.elapsed());
    }
    {
        let start = Instant::now();
        eprint!("trivial change: ");
        host.raw_database_mut().salsa_runtime_mut().synthetic_write(Durability::LOW);
        work(&host.analysis());
        eprintln!("{:?}", start.elapsed());
    }
    {
        let start = Instant::now();
        eprint!("comment change: ");
        {
            let mut text = host.analysis().file_text(file_id).unwrap().to_string();
            text.push_str("\n/* Hello world */\n");
            let mut change = Change::new();
            change.change_file(file_id, Some(Arc::new(text)));
            host.apply_change(change);
        }
        work(&host.analysis());
        eprintln!("{:?}", start.elapsed());
    }
    {
        let start = Instant::now();
        eprint!("item change:    ");
        {
            let mut text = host.analysis().file_text(file_id).unwrap().to_string();
            text.push_str("\npub fn _dummy() {}\n");
            let mut change = Change::new();
            change.change_file(file_id, Some(Arc::new(text)));
            host.apply_change(change);
        }
        work(&host.analysis());
        eprintln!("{:?}", start.elapsed());
    }
    {
        let start = Instant::now();
        eprint!("const change:   ");
        host.raw_database_mut().salsa_runtime_mut().synthetic_write(Durability::HIGH);
        let res = work(&host.analysis());
        eprintln!("{:?}", start.elapsed());
        res
    }
}
