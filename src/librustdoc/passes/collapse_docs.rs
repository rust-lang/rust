use crate::clean;
use crate::core::DocContext;
use crate::passes::Pass;

crate const COLLAPSE_DOCS: Pass = Pass {
    name: "collapse-docs",
    run: collapse_docs,
    description: "concatenates all document attributes into one document attribute",
};

crate fn collapse_docs(mut krate: clean::Crate, _: &DocContext<'_>) -> clean::Crate {
    krate.collapsed = true;
    krate
}
