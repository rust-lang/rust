use rustc_ast::{AttrItem, ForeignMod, ast};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_session::config::Offload;
use rustc_span::{DUMMY_SP, Ident, Span, sym};
use thin_vec::thin_vec;

use crate::errors;

/*
```
#[offload_kernel]
fn foo(..args) {
    // body
}
```

expands to:
```
#[cfg(host)]
unsafe extern "C" {
    pub fn foo(..args)
}

#[cfg(device)]
#[rustc_offload_kernel]
unsafe extern "gpu-kernel" fn foo(args) {
    // body
}
```
*/
fn compile_for_device(ecx: &mut ExtCtxt<'_>) -> bool {
    ecx.sess.opts.unstable_opts.offload.contains(&Offload::Device)
}

fn outer_normal_attr(
    kind: &Box<rustc_ast::NormalAttr>,
    id: rustc_ast::AttrId,
    span: Span,
) -> rustc_ast::Attribute {
    let style = rustc_ast::AttrStyle::Outer;
    let kind = rustc_ast::AttrKind::Normal(kind.clone());
    rustc_ast::Attribute { kind, id, style, span }
}

fn extract_fn(
    item: &Annotatable,
) -> Option<(ast::Visibility, ast::FnSig, Ident, ast::Generics, Option<Box<ast::Block>>)> {
    match item {
        Annotatable::Item(iitem) => match &iitem.kind {
            ast::ItemKind::Fn(ast::Fn { sig, ident, generics, body, .. }) => {
                Some((iitem.vis.clone(), sig.clone(), *ident, generics.clone(), body.clone()))
            }
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn expand_kernel(
    ecx: &mut ExtCtxt<'_>,
    expand_span: Span,
    _meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    let dcx = ecx.sess.dcx();

    let Some((vis, sig, ident, generics, body)) = extract_fn(&item) else {
        dcx.emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
        return vec![item];
    };

    let span = ecx.with_def_site_ctxt(expand_span);

    // device function
    let mut device_fn = Box::new(ast::Fn {
        defaultness: ast::Defaultness::Implicit,
        sig: sig.clone(),
        ident,
        generics: generics.clone(),
        contract: None,
        body,
        define_opaque: None,
        eii_impls: Default::default(),
    });

    let extern_gpu_kernel = ast::Extern::from_abi(
        Some(ast::StrLit {
            symbol: sym::gpu_kernel,
            suffix: None,
            symbol_unescaped: sym::gpu_kernel,
            style: ast::StrStyle::Cooked,
            span,
        }),
        span,
    );
    device_fn.sig.header.ext = extern_gpu_kernel;
    device_fn.sig.header.safety = ast::Safety::Unsafe(span);

    // rustc_offload_kernel attr
    let rustc_offload_kernel_attr =
        Box::new(ast::NormalAttr::from_ident(Ident::with_dummy_span(sym::rustc_offload_kernel)));
    let rustc_offload_kernel = outer_normal_attr(
        &rustc_offload_kernel_attr,
        ecx.sess.psess.attr_id_generator.mk_attr_id(),
        span,
    );

    // unsafe(no_mangle) attr
    let unsafe_item = AttrItem {
        unsafety: ast::Safety::Unsafe(span),
        path: ast::Path::from_ident(Ident::new(sym::no_mangle, span)),
        args: ast::AttrItemKind::Unparsed(ast::AttrArgs::Empty),
        tokens: None,
    };

    let no_mangle_attr = Box::new(ast::NormalAttr { item: unsafe_item, tokens: None });
    let new_id = ecx.sess.psess.attr_id_generator.mk_attr_id();
    let unsafe_no_mangle = outer_normal_attr(&no_mangle_attr, new_id, span);

    let device_item = {
        let mut item = ecx.item(
            span,
            thin_vec![rustc_offload_kernel, unsafe_no_mangle],
            ast::ItemKind::Fn(device_fn),
        );
        item.vis = vis.clone();
        Annotatable::Item(item)
    };

    // host function
    let host_fn = Box::new(ast::Fn {
        defaultness: ast::Defaultness::Implicit,
        sig: sig.clone(),
        ident,
        generics: generics.clone(),
        contract: None,
        body: None,
        define_opaque: None,
        eii_impls: Default::default(),
    });

    let foreign_fn = ast::ForeignItem {
        attrs: Default::default(),
        id: ast::DUMMY_NODE_ID,
        span,
        vis: vis.clone(),
        kind: ast::ForeignItemKind::Fn(host_fn),
        tokens: None,
    };

    let extern_c_lit = ast::StrLit {
        symbol: sym::C,
        suffix: None,
        symbol_unescaped: sym::C,
        style: ast::StrStyle::Cooked,
        span,
    };

    let foreign_mod = ForeignMod {
        abi: Some(extern_c_lit),
        safety: ast::Safety::Unsafe(span),
        items: thin_vec![Box::new(foreign_fn)],
        extern_span: DUMMY_SP,
    };

    let host_item = {
        let mut item = ecx.item(span, thin_vec![], ast::ItemKind::ForeignMod(foreign_mod));
        item.vis = vis;
        Annotatable::Item(item)
    };

    if compile_for_device(ecx) { vec![device_item] } else { vec![host_item] }
}
