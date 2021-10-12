use std::error::Error as StdError;

use crate::error::Error;

pub(crate) fn load() -> Result<tera::Tera, Error> {
    let mut templates = tera::Tera::default();

    macro_rules! include_template {
        ($file:literal, $fullpath:literal) => {
            templates.add_raw_template($file, include_str!($fullpath)).map_err(|e| Error {
                file: $file.into(),
                error: format!("{}: {}", e, e.source().map(|e| e.to_string()).unwrap_or_default()),
            })?
        };
    }

    include_template!("page.html", "../templates/page.html");
    include_template!("print_item.html", "../templates/print_item.html");
    Ok(templates)
}
