use crate::cache::Cache;
use crate::error::{DiagCtxt, Source};
use crate::{channel, Command, CommandKind};

impl Command<'_> {
    pub(crate) fn check(self, cache: &mut Cache<'_>, dcx: &mut DiagCtxt) -> Result<(), ()> {
        let result = self.kind.check(cache, self.source.clone(), dcx)?;

        if result == self.negated {
            // FIXME: better diag
            dcx.emit("check failed", self.source, None);
            return Err(());
        }

        Ok(())
    }
}

impl CommandKind {
    // FIXME: implement all checks!
    fn check(
        self,
        cache: &mut Cache<'_>,
        _source: Source<'_>, // FIXME: unused
        dcx: &mut DiagCtxt,
    ) -> Result<bool, ()> {
        Ok(match self {
            Self::HasFile { path } => cache.has(path, dcx)?, // FIXME: check if it's actually a file
            Self::HasDir { path } => cache.has(path, dcx)?, // FIXME: check if it's actually a directory
            Self::Has { path, xpath, text } => {
                let _data = cache.load(path, dcx)?;
                _ = xpath;
                _ = text;
                true // FIXME
            }
            Self::HasRaw { path, text } => {
                let data = cache.load(path, dcx)?;

                if text.is_empty() {
                    // fast path
                    return Ok(true);
                }

                let text = channel::instantiate(&text, dcx)?;
                let text = text.replace(|c: char| c.is_ascii_whitespace(), " ");
                let data = data.replace(|c: char| c.is_ascii_whitespace(), " ");

                data.contains(&text)
            }
            Self::Matches { path, xpath, pattern } => {
                let _data = cache.load(path, dcx)?;
                _ = xpath;
                _ = pattern;

                true // FIXME
            }
            Self::MatchesRaw { path, pattern } => pattern.is_match(cache.load(path, dcx)?),
            Self::Count { path, xpath, text, count } => {
                let _data = cache.load(path, dcx)?;
                _ = xpath;
                _ = text;
                _ = count;
                true // FIXME
            }
            Self::Files { path, files } => {
                let _data = cache.load(path, dcx)?;
                _ = files;
                true // FIXME
            }
            Self::Snapshot { name, path, xpath } => {
                let _data = cache.load(path, dcx)?;
                _ = name;
                _ = path;
                _ = xpath;
                true // FIXME
            }
        })
    }
}
