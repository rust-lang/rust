//! Benchmark operations like highlighting or goto definition.

use std::{env, path::Path, str::FromStr, sync::Arc, time::Instant};

use anyhow::{format_err, Result};
use ra_db::{
    salsa::{Database, Durability},
    AbsPathBuf, FileId,
};
use ra_ide::{Analysis, AnalysisChange, AnalysisHost, CompletionConfig, FilePosition, LineCol};

use crate::cli::{load_cargo::load_cargo, Verbosity};

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
        let (path_line, column) = rsplit_at_char(s, ':')?;
        let (path, line) = rsplit_at_char(path_line, ':')?;
        let path = env::current_dir().unwrap().join(path);
        let path = AbsPathBuf::assert(path);
        Ok(Position { path, line: line.parse()?, column: column.parse()? })
    }
}

fn rsplit_at_char(s: &str, c: char) -> Result<(&str, &str)> {
    let idx = s.rfind(c).ok_or_else(|| format_err!("no `{}` in {}", c, s))?;
    Ok((&s[..idx], &s[idx + 1..]))
}

pub fn analysis_bench(
    verbosity: Verbosity,
    path: &Path,
    what: BenchWhat,
    load_output_dirs: bool,
    with_proc_macro: bool,
) -> Result<()> {
    ra_prof::init();

    let start = Instant::now();
    eprint!("loading: ");
    let (mut host, vfs) = load_cargo(path, load_output_dirs, with_proc_macro)?;
    eprintln!("{:?}\n", start.elapsed());

    let file_id = {
        let path = match &what {
            BenchWhat::Highlight { path } => path,
            BenchWhat::Complete(pos) | BenchWhat::GotoDef(pos) => &pos.path,
        };
        let path = path.clone().into();
        vfs.file_id(&path).ok_or_else(|| format_err!("Can't find {}", path))?
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
            let is_completion = matches!(what, BenchWhat::Complete(..));

            let offset = host
                .analysis()
                .file_line_index(file_id)?
                .offset(LineCol { line: pos.line - 1, col_utf16: pos.column });
            let file_position = FilePosition { file_id, offset };

            if is_completion {
                let options = CompletionConfig::default();
                let res = do_work(&mut host, file_id, |analysis| {
                    analysis.completions(&options, file_position)
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
            let mut change = AnalysisChange::new();
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
