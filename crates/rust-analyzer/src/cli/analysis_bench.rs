//! Benchmark operations like highlighting or goto definition.

use std::{
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Instant,
};

use anyhow::{format_err, Result};
use ra_db::{
    salsa::{Database, Durability},
    FileId, SourceDatabaseExt,
};
use ra_ide::{Analysis, AnalysisChange, AnalysisHost, CompletionOptions, FilePosition, LineCol};

use crate::cli::{load_cargo::load_cargo, Verbosity};

pub enum BenchWhat {
    Highlight { path: PathBuf },
    Complete(Position),
    GotoDef(Position),
}

pub struct Position {
    pub path: PathBuf,
    pub line: u32,
    pub column: u32,
}

impl FromStr for Position {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let (path_line, column) = rsplit_at_char(s, ':')?;
        let (path, line) = rsplit_at_char(path_line, ':')?;
        Ok(Position { path: path.into(), line: line.parse()?, column: column.parse()? })
    }
}

fn rsplit_at_char(s: &str, c: char) -> Result<(&str, &str)> {
    let idx = s.rfind(c).ok_or_else(|| format_err!("no `{}` in {}", c, s))?;
    Ok((&s[..idx], &s[idx + 1..]))
}

pub fn analysis_bench(verbosity: Verbosity, path: &Path, what: BenchWhat) -> Result<()> {
    ra_prof::init();

    let start = Instant::now();
    eprint!("loading: ");
    let (mut host, roots) = load_cargo(path)?;
    let db = host.raw_database();
    eprintln!("{:?}\n", start.elapsed());

    let file_id = {
        let path = match &what {
            BenchWhat::Highlight { path } => path,
            BenchWhat::Complete(pos) | BenchWhat::GotoDef(pos) => &pos.path,
        };
        let path = std::env::current_dir()?.join(path).canonicalize()?;
        roots
            .iter()
            .find_map(|(source_root_id, project_root)| {
                if project_root.is_member() {
                    for file_id in db.source_root(*source_root_id).walk() {
                        let rel_path = db.file_relative_path(file_id);
                        let abs_path = rel_path.to_path(project_root.path());
                        if abs_path == path {
                            return Some(file_id);
                        }
                    }
                }
                None
            })
            .ok_or_else(|| format_err!("Can't find {}", path.display()))?
    };

    match &what {
        BenchWhat::Highlight { .. } => {
            let res = do_work(&mut host, file_id, |analysis| {
                analysis.diagnostics(file_id).unwrap();
                analysis.highlight_as_html(file_id, false).unwrap()
            });
            if verbosity.is_verbose() {
                println!("\n{}", res);
            }
        }
        BenchWhat::Complete(pos) | BenchWhat::GotoDef(pos) => {
            let is_completion = match what {
                BenchWhat::Complete(..) => true,
                _ => false,
            };

            let offset = host
                .analysis()
                .file_line_index(file_id)?
                .offset(LineCol { line: pos.line - 1, col_utf16: pos.column });
            let file_position = FilePosition { file_id, offset };

            if is_completion {
                let options = CompletionOptions::default();
                let res = do_work(&mut host, file_id, |analysis| {
                    analysis.completions(file_position, &options)
                });
                if verbosity.is_verbose() {
                    println!("\n{:#?}", res);
                }
            } else {
                let res =
                    do_work(&mut host, file_id, |analysis| analysis.goto_definition(file_position));
                if verbosity.is_verbose() {
                    println!("\n{:#?}", res);
                }
            }
        }
    }
    Ok(())
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
            let mut change = AnalysisChange::new();
            change.change_file(file_id, Arc::new(text));
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
