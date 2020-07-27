use crate::{ast, attr, visit};
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;

#[derive(Clone, Copy)]
pub enum AllocatorKind {
    Global,
    Default,
}

impl AllocatorKind {
    pub fn fn_name(&self, base: Symbol) -> String {
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
    pub name: Symbol,
    pub inputs: &'static [AllocatorTy],
    pub output: AllocatorTy,
}

pub static ALLOCATOR_METHODS: &[AllocatorMethod] = &[
    AllocatorMethod {
        name: sym::alloc,
        inputs: &[AllocatorTy::Layout],
        output: AllocatorTy::ResultPtr,
    },
    AllocatorMethod {
        name: sym::dealloc,
        inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout],
        output: AllocatorTy::Unit,
    },
    AllocatorMethod {
        name: sym::realloc,
        inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout, AllocatorTy::Usize],
        output: AllocatorTy::ResultPtr,
    },
    AllocatorMethod {
        name: sym::alloc_zeroed,
        inputs: &[AllocatorTy::Layout],
        output: AllocatorTy::ResultPtr,
    },
];

pub fn global_allocator_spans(krate: &ast::Crate) -> Vec<Span> {
    struct Finder {
        name: Symbol,
        spans: Vec<Span>,
    }
    impl<'ast> visit::Visitor<'ast> for Finder {
        fn visit_item(&mut self, item: &'ast ast::Item) {
            if item.ident.name == self.name
                && attr::contains_name(&item.attrs, sym::rustc_std_internal_symbol)
            {
                self.spans.push(item.span);
            }
            visit::walk_item(self, item)
        }
    }

    let name = Symbol::intern(&AllocatorKind::Global.fn_name(sym::alloc));
    let mut f = Finder { name, spans: Vec::new() };
    visit::walk_crate(&mut f, krate);
    f.spans
}
