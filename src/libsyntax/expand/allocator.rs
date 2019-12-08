use crate::{ast, attr, visit};
use syntax_pos::symbol::{sym, Symbol};
use syntax_pos::Span;

#[derive(Clone, Copy)]
pub enum AllocatorKind {
    Global,
    Default,
}

impl AllocatorKind {
    pub fn fn_name(&self, base: &str) -> String {
        match *self {
            AllocatorKind::Global => format!("__rg_{}", base),
            AllocatorKind::Default => format!("__rdl_{}", base),
        }
    }
}

pub enum AllocatorTy {
    Layout,
    Ptr,
    ResultPtr,
    Unit,
    Usize,
}

pub struct AllocatorMethod {
    pub name: &'static str,
    pub inputs: &'static [AllocatorTy],
    pub output: AllocatorTy,
}

pub static ALLOCATOR_METHODS: &[AllocatorMethod] = &[
    AllocatorMethod {
        name: "alloc",
        inputs: &[AllocatorTy::Layout],
        output: AllocatorTy::ResultPtr,
    },
    AllocatorMethod {
        name: "dealloc",
        inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout],
        output: AllocatorTy::Unit,
    },
    AllocatorMethod {
        name: "realloc",
        inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout, AllocatorTy::Usize],
        output: AllocatorTy::ResultPtr,
    },
    AllocatorMethod {
        name: "alloc_zeroed",
        inputs: &[AllocatorTy::Layout],
        output: AllocatorTy::ResultPtr,
    },
];

pub fn global_allocator_spans(krate: &ast::Crate) -> Vec<Span> {
    struct Finder { name: Symbol, spans: Vec<Span> }
    impl<'ast> visit::Visitor<'ast> for Finder {
        fn visit_item(&mut self, item: &'ast ast::Item) {
            if item.ident.name == self.name &&
               attr::contains_name(&item.attrs, sym::rustc_std_internal_symbol) {
                self.spans.push(item.span);
            }
            visit::walk_item(self, item)
        }
    }

    let name = Symbol::intern(&AllocatorKind::Global.fn_name("alloc"));
    let mut f = Finder { name, spans: Vec::new() };
    visit::walk_crate(&mut f, krate);
    f.spans
}
