use std::collections::HashMap;

pub(crate) struct LicensesInterner {
    by_id: Vec<License>,
    by_struct: HashMap<License, usize>,
}

impl LicensesInterner {
    pub(crate) fn new() -> Self {
        LicensesInterner { by_id: Vec::new(), by_struct: HashMap::new() }
    }

    pub(crate) fn intern(&mut self, license: License) -> LicenseId {
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
