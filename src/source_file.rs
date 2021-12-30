use std::fs;
use std::io::{self, Write};
use std::path::Path;

use crate::config::FileName;
use crate::emitter::{self, Emitter};
use crate::parse::session::ParseSess;
use crate::NewlineStyle;

#[cfg(test)]
use crate::config::Config;
#[cfg(test)]
use crate::create_emitter;
#[cfg(test)]
use crate::formatting::FileRecord;

use rustc_data_structures::sync::Lrc;

// Append a newline to the end of each file.
pub(crate) fn append_newline(s: &mut String) {
    s.push('\n');
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
    let mut emitter = create_emitter(config);

    emitter.emit_header(out)?;
    for &(ref filename, ref text) in source_file {
        write_file(
            None,
            filename,
            text,
            out,
            &mut *emitter,
            config.newline_style(),
        )?;
    }
    emitter.emit_footer(out)?;

    Ok(())
}

pub(crate) fn write_file<T>(
    parse_sess: Option<&ParseSess>,
    filename: &FileName,
    formatted_text: &str,
    out: &mut T,
    emitter: &mut dyn Emitter,
    newline_style: NewlineStyle,
) -> Result<emitter::EmitterResult, io::Error>
where
    T: Write,
{
    fn ensure_real_path(filename: &FileName) -> &Path {
        match *filename {
            FileName::Real(ref path) => path,
            _ => panic!("cannot format `{}` and emit to files", filename),
        }
    }

    impl From<&FileName> for rustc_span::FileName {
        fn from(filename: &FileName) -> rustc_span::FileName {
            match filename {
                FileName::Real(path) => {
                    rustc_span::FileName::Real(rustc_span::RealFileName::LocalPath(path.to_owned()))
                }
                FileName::Stdin => rustc_span::FileName::Custom("stdin".to_owned()),
            }
        }
    }

    // SourceFile's in the SourceMap will always have Unix-style line endings
    // See: https://github.com/rust-lang/rustfmt/issues/3850
    // So if the user has explicitly overridden the rustfmt `newline_style`
    // config and `filename` is FileName::Real, then we must check the file system
    // to get the original file value in order to detect newline_style conflicts.
    // Otherwise, parse session is around (cfg(not(test))) and newline_style has been
    // left as the default value, then try getting source from the parse session
    // source map instead of hitting the file system. This also supports getting
    // original text for `FileName::Stdin`.
    let original_text = if newline_style != NewlineStyle::Auto && *filename != FileName::Stdin {
        Lrc::new(fs::read_to_string(ensure_real_path(filename))?)
    } else {
        match parse_sess.and_then(|sess| sess.get_original_snippet(filename)) {
            Some(ori) => ori,
            None => Lrc::new(fs::read_to_string(ensure_real_path(filename))?),
        }
    };

    let formatted_file = emitter::FormattedFile {
        filename,
        original_text: original_text.as_str(),
        formatted_text,
    };

    emitter.emit_formatted_file(out, formatted_file)
}
