use std::fs;
use std::io::{self, Write};
use std::path::Path;

use syntax::source_map::SourceMap;

use crate::config::FileName;
use crate::emitter::{self, Emitter};

#[cfg(test)]
use crate::config::Config;
#[cfg(test)]
use crate::create_emitter;
#[cfg(test)]
use crate::formatting::FileRecord;

// Append a newline to the end of each file.
pub(crate) fn append_newline(s: &mut String) {
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
    let emitter = create_emitter(config);

    emitter.emit_header(out)?;
    for &(ref filename, ref text) in source_file {
        write_file(None, filename, text, out, &*emitter)?;
    }
    emitter.emit_footer(out)?;

    Ok(())
}

pub(crate) fn write_file<T>(
    source_map: Option<&SourceMap>,
    filename: &FileName,
    formatted_text: &str,
    out: &mut T,
    emitter: &dyn Emitter,
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
        .and_then(|x| x.src.as_ref().map(ToString::to_string));
    let original_text = match original_text {
        Some(ori) => ori,
        None => fs::read_to_string(ensure_real_path(filename))?,
    };

    let formatted_file = emitter::FormattedFile {
        filename,
        original_text: &original_text,
        formatted_text,
    };

    emitter.emit_formatted_file(out, formatted_file)
}
