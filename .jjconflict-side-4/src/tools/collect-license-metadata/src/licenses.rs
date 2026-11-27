use std::collections::HashMap;

const COPYRIGHT_PREFIXES: &[&str] = &["SPDX-FileCopyrightText:", "Copyright", "(c)", "(C)", "©"];

pub(crate) struct LicensesInterner {
    by_id: Vec<License>,
    by_struct: HashMap<License, usize>,
}

impl LicensesInterner {
    pub(crate) fn new() -> Self {
        LicensesInterner { by_id: Vec::new(), by_struct: HashMap::new() }
    }

    pub(crate) fn intern(&mut self, mut license: License) -> LicenseId {
        license.simplify();
        if let Some(id) = self.by_struct.get(&license) {
            LicenseId(*id)
        } else {
            let id = self.by_id.len();
            self.by_id.push(license.clone());
            self.by_struct.insert(license, id);
            LicenseId(id)
        }
    }

    pub(crate) fn resolve(&self, id: LicenseId) -> &License {
        &self.by_id[id.0]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(transparent)]
pub(crate) struct LicenseId(usize);

#[derive(Clone, Hash, PartialEq, Eq, serde::Serialize)]
pub(crate) struct License {
    pub(crate) spdx: String,
    pub(crate) copyright: Vec<String>,
}

impl License {
    fn simplify(&mut self) {
        self.remove_copyright_prefixes();
        self.remove_trailing_dots();
        self.copyright.sort();
        self.copyright.dedup();
    }

    fn remove_copyright_prefixes(&mut self) {
        for copyright in &mut self.copyright {
            let mut stripped = copyright.trim();
            let mut previous_stripped;
            loop {
                previous_stripped = stripped;
                for pattern in COPYRIGHT_PREFIXES {
                    stripped = stripped.trim_start_matches(pattern).trim_start();
                }
                if stripped == previous_stripped {
                    break;
                }
            }
            *copyright = stripped.into();
        }
    }

    fn remove_trailing_dots(&mut self) {
        for copyright in &mut self.copyright {
            if copyright.ends_with('.') {
                *copyright = copyright.trim_end_matches('.').to_string();
            }
        }
    }
}
