//! FIXME: write short doc here

use std::{
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Instant,
};

use ra_db::{
    salsa::{Database, Durability},
    FileId, SourceDatabaseExt,
};
use ra_ide::{Analysis, AnalysisChange, AnalysisHost, FilePosition, LineCol};

use crate::{load_cargo::load_cargo, Result};

pub(crate) struct Position {
    path: PathBuf,
    line: u32,
    column: u32,
}

impl FromStr for Position {
    type Err = Box<dyn std::error::Error + Send + Sync>;
    fn from_str(s: &str) -> Result<Self> {
        let (path_line, column) = rsplit_at_char(s, ':')?;
        let (path, line) = rsplit_at_char(path_line, ':')?;
        Ok(Position { path: path.into(), line: line.parse()?, column: column.parse()? })
    }
}

fn rsplit_at_char(s: &str, c: char) -> Result<(&str, &str)> {
    let idx = s.rfind(':').ok_or_else(|| format!("no `{}` in {}", c, s))?;
    Ok((&s[..idx], &s[idx + 1..]))
}

pub(crate) enum Op {
    Highlight { path: PathBuf },
    Complete(Position),
    GotoDef(Position),
}

pub(crate) fn run(verbose: bool, path: &Path, op: Op) -> Result<()> {
    ra_prof::init();

    let start = Instant::now();
    eprint!("loading: ");
    let (mut host, roots) = load_cargo(path)?;
    let db = host.raw_database();
    eprintln!("{:?}\n", start.elapsed());

    let file_id = {
        let path = match &op {
            Op::Highlight { path } => path,
            Op::Complete(pos) | Op::GotoDef(pos) => &pos.path,
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
            .ok_or_else(|| format!("Can't find {:?}", path))?
    };

    match &op {
        Op::Highlight { .. } => {
            let res = do_work(&mut host, file_id, |analysis| {
                analysis.diagnostics(file_id).unwrap();
                analysis.highlight_as_html(file_id, false).unwrap()
            });
            if verbose {
                println!("\n{}", res);
            }
        }
        Op::Complete(pos) | Op::GotoDef(pos) => {
            let is_completion = match op {
                Op::Complete(..) => true,
                _ => false,
            };

            let offset = host
                .analysis()
                .file_line_index(file_id)?
                .offset(LineCol { line: pos.line - 1, col_utf16: pos.column });
            let file_postion = FilePosition { file_id, offset };

            if is_completion {
                let res =
                    do_work(&mut host, file_id, |analysis| analysis.completions(file_postion));
                if verbose {
                    println!("\n{:#?}", res);
                }
            } else {
                let res =
                    do_work(&mut host, file_id, |analysis| analysis.goto_definition(file_postion));
                if verbose {
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
