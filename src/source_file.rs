use std::fs;
use std::io::{self, Write};
use std::path::Path;

use syntax::source_map::SourceMap;

use crate::checkstyle::output_checkstyle_file;
use crate::config::{Config, EmitMode, FileName, Verbosity};
use crate::rustfmt_diff::{make_diff, print_diff, ModifiedLines};

#[cfg(test)]
use crate::formatting::FileRecord;

// Append a newline to the end of each file.
pub fn append_newline(s: &mut String) {
    s.push_str("\n");
}

#[cfg(test)]
pub(crate) fn write_all_files<T>(
    source_file: &[FileRecord],
    out: &mut T,
    config: &Config,
) -> Result<(), io::Error>
where
    T: Write,
{
    if config.emit_mode() == EmitMode::Checkstyle {
        write!(out, "{}", crate::checkstyle::header())?;
    }
    for &(ref filename, ref text) in source_file {
        write_file(None, filename, text, out, config)?;
    }
    if config.emit_mode() == EmitMode::Checkstyle {
        write!(out, "{}", crate::checkstyle::footer())?;
    }

    Ok(())
}

pub fn write_file<T>(
    source_map: Option<&SourceMap>,
    filename: &FileName,
    formatted_text: &str,
    out: &mut T,
    config: &Config,
) -> Result<bool, io::Error>
where
    T: Write,
{
    fn ensure_real_path(filename: &FileName) -> &Path {
        match *filename {
            FileName::Real(ref path) => path,
            _ => panic!("cannot format `{}` and emit to files", filename),
        }
    }

    impl From<&FileName> for syntax_pos::FileName {
        fn from(filename: &FileName) -> syntax_pos::FileName {
            match filename {
                FileName::Real(path) => syntax_pos::FileName::Real(path.to_owned()),
                FileName::Stdin => syntax_pos::FileName::Custom("stdin".to_owned()),
            }
        }
    }

    // If parse session is around (cfg(not(test))) then try getting source from
    // there instead of hitting the file system. This also supports getting
    // original text for `FileName::Stdin`.
    let original_text = source_map
        .and_then(|x| x.get_source_file(&filename.into()))
        .and_then(|x| x.src.as_ref().map(|x| x.to_string()));
    let original_text = match original_text {
        Some(ori) => ori,
        None => fs::read_to_string(ensure_real_path(filename))?,
    };

    match config.emit_mode() {
        EmitMode::Files if config.make_backup() => {
            let filename = ensure_real_path(filename);
            if original_text != formatted_text {
                // Do a little dance to make writing safer - write to a temp file
                // rename the original to a .bk, then rename the temp file to the
                // original.
                let tmp_name = filename.with_extension("tmp");
                let bk_name = filename.with_extension("bk");

                fs::write(&tmp_name, formatted_text)?;
                fs::rename(filename, bk_name)?;
                fs::rename(tmp_name, filename)?;
            }
        }
        EmitMode::Files => {
            // Write text directly over original file if there is a diff.
            let filename = ensure_real_path(filename);

            if original_text != formatted_text {
                fs::write(filename, formatted_text)?;
            }
        }
        EmitMode::Stdout | EmitMode::Coverage => {
            if config.verbose() != Verbosity::Quiet {
                println!("{}:\n", filename);
            }
            write!(out, "{}", formatted_text)?;
        }
        EmitMode::ModifiedLines => {
            let mismatch = make_diff(&original_text, formatted_text, 0);
            let has_diff = !mismatch.is_empty();
            write!(out, "{}", ModifiedLines::from(mismatch))?;
            return Ok(has_diff);
        }
        EmitMode::Checkstyle => {
            let filename = ensure_real_path(filename);

            let diff = make_diff(&original_text, formatted_text, 3);
            output_checkstyle_file(out, filename, diff)?;
        }
        EmitMode::Diff => {
            let mismatch = make_diff(&original_text, formatted_text, 3);
            let has_diff = !mismatch.is_empty();
            print_diff(
                mismatch,
                |line_num| format!("Diff in {} at line {}:", filename, line_num),
                config,
            );
            return Ok(has_diff);
        }
    }

    // when we are not in diff mode, don't indicate differing files
    Ok(false)
}
