//! Tidy check to verify the validity of long error diagnostic codes.
//!
//! This ensures that error codes are used at most once and also prints out some
//! statistics about the error codes.

use std::collections::HashMap;
use std::path::Path;

pub fn check(path: &Path, bad: &mut bool) {
    let mut map: HashMap<_, Vec<_>> = HashMap::new();
    super::walk(path,
                &mut |path| super::filter_dirs(path) || path.ends_with("src/test"),
                &mut |entry, contents| {
        let file = entry.path();
        let filename = file.file_name().unwrap().to_string_lossy();
        if filename != "error_codes.rs" {
            return
        }

        // In the `register_long_diagnostics!` macro, entries look like this:
        //
        // ```
        // EXXXX: r##"
        // <Long diagnostic message>
        // "##,
        // ```
        //
        // and these long messages often have error codes themselves inside
        // them, but we don't want to report duplicates in these cases. This
        // variable keeps track of whether we're currently inside one of these
        // long diagnostic messages.
        let mut inside_long_diag = false;
        for (num, line) in contents.lines().enumerate() {
            if inside_long_diag {
                inside_long_diag = !line.contains("\"##");
                continue
            }

            let mut search = line;
            while let Some(i) = search.find('E') {
                search = &search[i + 1..];
                let code = if search.len() > 4 {
                    search[..4].parse::<u32>()
                } else {
                    continue
                };
                let code = match code {
                    Ok(n) => n,
                    Err(..) => continue,
                };
                map.entry(code).or_default()
                   .push((file.to_owned(), num + 1, line.to_owned()));
                break
            }

            inside_long_diag = line.contains("r##\"");
        }
    });

    let mut max = 0;
    for (&code, entries) in map.iter() {
        if code > max {
            max = code;
        }
        if entries.len() == 1 {
            continue
        }

        tidy_error!(bad, "duplicate error code: {}", code);
        for &(ref file, line_num, ref line) in entries.iter() {
            tidy_error!(bad, "{}:{}: {}", file.display(), line_num, line);
        }
    }

    if !*bad {
        println!("* {} error codes", map.len());
        println!("* highest error code: E{:04}", max);
    }
}
