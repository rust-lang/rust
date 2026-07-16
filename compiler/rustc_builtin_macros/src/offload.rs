use rustc_ast::token::{Delimiter, Token, TokenKind};
use rustc_ast::tokenstream::{DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_ast::{AttrItem, ast};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_session::config::Offload;
use rustc_span::{DUMMY_SP, Ident, Span, sym};
use thin_vec::thin_vec;

use crate::diagnostics;

fn compile_for_device(ecx: &mut ExtCtxt<'_>) -> bool {
    ecx.sess.opts.unstable_opts.offload.contains(&Offload::Device)
}

fn outer_normal_attr(normal: &Box<ast::NormalAttr>, id: ast::AttrId, span: Span) -> ast::Attribute {
    let style = ast::AttrStyle::Outer;
    let kind = ast::AttrKind::Normal(normal.clone());
    ast::Attribute { kind, id, style, span }
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

/// The `offload_kernel` macro expands the function into two separate definitions:
/// one on the host to handle the call, and one on the device for executing the kernel.
///
/// ```
/// #[offload_kernel]
/// fn foo(a: &[f32], b: &[f32], c: *mut f32) {
///     *c = a[0] + b[0];
/// }
/// ```
///
/// This expands to the host-side function:
///
/// ```
/// #[unsafe(no_mangle)]
/// #[inline(never)]
/// fn foo(_: &[f32], _: &[f32], _: *mut f32) {
///     ::core::panicking::panic("not implemented")
/// }
/// ```
///
/// And the device-side kernel:
///
/// ```
/// #[rustc_offload_kernel]
/// #[unsafe(no_mangle)]
/// unsafe extern "gpu-kernel" fn foo(a: &[f32], b: &[f32], c: *mut f32) {
///     *c = a[0] + b[0];
/// }
/// ```
pub(crate) fn expand_kernel(
    ecx: &mut ExtCtxt<'_>,
    expand_span: Span,
    _meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    let dcx = ecx.sess.dcx();

    let Some((vis, sig, ident, generics, body)) = extract_fn(&item) else {
        dcx.emit_err(diagnostics::AutoDiffInvalidApplication { span: item.span() });
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
        args: ast::AttrArgs::Empty,
        span,
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

    // unimplemented! body
    let macro_expr = ecx.expr_macro_call(
        span,
        ecx.macro_call(
            span,
            ecx.path_global(
                span,
                [sym::std, sym::unimplemented].map(|s| Ident::new(s, span)).to_vec(),
            ),
            Delimiter::Parenthesis,
            TokenStream::default(),
        ),
    );
    let stmt = ecx.stmt_expr(macro_expr);
    let body = ecx.block(span, thin_vec![stmt]);

    // host function
    let mut host_fn = Box::new(ast::Fn {
        defaultness: ast::Defaultness::Implicit,
        sig: sig.clone(),
        ident,
        generics: generics.clone(),
        contract: None,
        body: Some(body),
        define_opaque: None,
        eii_impls: Default::default(),
    });

    for param in host_fn.sig.decl.inputs.iter_mut() {
        param.pat = Box::new(ecx.pat_wild(param.pat.span));
    }

    // inline(never) attr
    let ts: Vec<TokenTree> = vec![TokenTree::Token(
        Token::new(TokenKind::Ident(sym::never, false.into()), span),
        Spacing::Joint,
    )];

    let never_arg = ast::DelimArgs {
        dspan: DelimSpan::from_single(span),
        delim: Delimiter::Parenthesis,
        tokens: TokenStream::from_iter(ts),
    };

    let inline_item = ast::AttrItem {
        unsafety: ast::Safety::Default,
        path: ast::Path::from_ident(Ident::with_dummy_span(sym::inline)),
        args: ast::AttrArgs::Delimited(never_arg),
        span: DUMMY_SP,
    };
    let inline_never_attr = Box::new(ast::NormalAttr { item: inline_item, tokens: None });

    let new_id = ecx.sess.psess.attr_id_generator.mk_attr_id();
    let inline_never = outer_normal_attr(&inline_never_attr, new_id, span);

    let new_id = ecx.sess.psess.attr_id_generator.mk_attr_id();
    let unsafe_no_mangle = outer_normal_attr(&no_mangle_attr, new_id, span);

    let host_item = {
        let mut item =
            ecx.item(span, thin_vec![unsafe_no_mangle, inline_never], ast::ItemKind::Fn(host_fn));
        item.vis = vis.clone();
        Annotatable::Item(item)
    };

    if compile_for_device(ecx) { vec![device_item] } else { vec![host_item] }
}
