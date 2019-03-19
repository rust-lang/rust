use crate::clean::*;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::{look_for_tests, Pass};

pub const CHECK_PRIVATE_ITEMS_DOC_TESTS: Pass = Pass {
    name: "check-private-items-doc-tests",
    pass: check_private_items_doc_tests,
    description: "check private items doc tests",
};

struct PrivateItemDocTestLinter<'a, 'tcx> {
    cx: &'a DocContext<'tcx>,
}

impl<'a, 'tcx> PrivateItemDocTestLinter<'a, 'tcx> {
    fn new(cx: &'a DocContext<'tcx>) -> Self {
        PrivateItemDocTestLinter {
            cx,
        }
    }
}

pub fn check_private_items_doc_tests(krate: Crate, cx: &DocContext<'_>) -> Crate {
    let mut coll = PrivateItemDocTestLinter::new(cx);

    coll.fold_crate(krate)
}

impl<'a, 'tcx> DocFolder for PrivateItemDocTestLinter<'a, 'tcx> {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        let cx = self.cx;
        let dox = item.attrs.collapsed_doc_value().unwrap_or_else(String::new);

        look_for_tests(&cx, &dox, &item, false);

        self.fold_item_recur(item)
    }
}
