use rustc_ast::tokenstream::TokenStream;
use rustc_expand::base::ExtCtxt;
use rustc_middle::ty::{TyCtxt, tls};
use rustc_proc_macro as pm;
use rustc_span::LocalExpnId;

type DeriveClient = pm::bridge::client::Client;

/// Stores the context necessary to expand a derive proc macro via a query.
struct QueryDeriveExpandCtx {
    /// Type-erased version of `&mut ExtCtxt`
    expansion_ctx: *mut (),
    client: DeriveClient,
}

impl QueryDeriveExpandCtx {
    /// Store the extension context and the client into the thread local value.
    /// It will be accessible via the `with` method while `f` is active.
    fn enter<F, R>(ecx: &mut ExtCtxt<'_>, client: DeriveClient, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // We need erasure to get rid of the lifetime
        let ctx = Self { expansion_ctx: ecx as *mut _ as *mut (), client };
        DERIVE_EXPAND_CTX.set(&ctx, f)
    }

    /// Accesses the thread local value of the derive expansion context.
    /// Must be called while the `enter` function is active.
    fn with<F, R>(f: F) -> R
    where
        F: for<'a, 'b> FnOnce(&'b mut ExtCtxt<'a>, DeriveClient) -> R,
    {
        DERIVE_EXPAND_CTX.with(|ctx| {
            let ectx = {
                let casted = ctx.expansion_ctx.cast::<ExtCtxt<'_>>();
                // SAFETY: We can only get the value from `with` while the `enter` function
                // is active (on the callstack), and that function's signature ensures that the
                // lifetime is valid.
                // If `with` is called at some other time, it will panic due to usage of
                // `scoped_tls::with`.
                unsafe { casted.as_mut().unwrap() }
            };

            f(ectx, ctx.client)
        })
    }
}

// When we invoke a query to expand a derive proc macro, we need to provide it with the expansion
// context and derive Client. We do that using a thread-local.
scoped_tls::scoped_thread_local!(static DERIVE_EXPAND_CTX: QueryDeriveExpandCtx);

pub(crate) fn expand_derive_macro(
    invoc_id: LocalExpnId,
    input: TokenStream,
    ecx: &mut ExtCtxt<'_>,
    client: DeriveClient,
) -> Result<TokenStream, ()> {
    tls::with(|tcx| {
        let input = &*tcx.arena.alloc(input);
        let key: (LocalExpnId, &TokenStream) = (invoc_id, input);

        QueryDeriveExpandCtx::enter(ecx, client, move || tcx.derive_macro_expansion(key).cloned())
    })
}

/// Provide a query for computing the output of a derive macro.
pub(crate) fn derive_macro_expansion<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (LocalExpnId, &'tcx TokenStream),
) -> Result<&'tcx TokenStream, ()> {
    let (invoc_id, input) = key;

    // Make sure that we invalidate the query when the crate defining the proc macro changes
    let _ = tcx.crate_hash(invoc_id.expn_data().macro_def_id.unwrap().krate);

    QueryDeriveExpandCtx::with(|ecx, client| {
        rustc_expand::proc_macro::expand_derive_macro(invoc_id, input.clone(), ecx, client)
            .map(|ts| &*tcx.arena.alloc(ts))
    })
}
