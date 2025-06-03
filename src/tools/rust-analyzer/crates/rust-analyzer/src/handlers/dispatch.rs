//! See [RequestDispatcher].
use std::{
    fmt::{self, Debug},
    panic, thread,
};

use ide_db::base_db::{
    DbPanicContext,
    salsa::{self, Cancelled, UnexpectedCycle},
};
use lsp_server::{ExtractError, Response, ResponseError};
use serde::{Serialize, de::DeserializeOwned};
use stdx::thread::ThreadIntent;

use crate::{
    global_state::{GlobalState, GlobalStateSnapshot},
    lsp::LspError,
    main_loop::Task,
    version::version,
};

/// A visitor for routing a raw JSON request to an appropriate handler function.
///
/// Most requests are read-only and async and are handled on the threadpool
/// (`on` method).
///
/// Some read-only requests are latency sensitive, and are immediately handled
/// on the main loop thread (`on_sync`). These are typically typing-related
/// requests.
///
/// Some requests modify the state, and are run on the main thread to get
/// `&mut` (`on_sync_mut`).
///
/// Read-only requests are wrapped into `catch_unwind` -- they don't modify the
/// state, so it's OK to recover from their failures.
pub(crate) struct RequestDispatcher<'a> {
    pub(crate) req: Option<lsp_server::Request>,
    pub(crate) global_state: &'a mut GlobalState,
}

impl RequestDispatcher<'_> {
    /// Dispatches the request onto the current thread, given full access to
    /// mutable global state. Unlike all other methods here, this one isn't
    /// guarded by `catch_unwind`, so, please, don't make bugs :-)
    pub(crate) fn on_sync_mut<R>(
        &mut self,
        f: fn(&mut GlobalState, R::Params) -> anyhow::Result<R::Result>,
    ) -> &mut Self
    where
        R: lsp_types::request::Request,
        R::Params: DeserializeOwned + panic::UnwindSafe + fmt::Debug,
        R::Result: Serialize,
    {
        let (req, params, panic_context) = match self.parse::<R>() {
            Some(it) => it,
            None => return self,
        };
        let _guard =
            tracing::info_span!("request", method = ?req.method, "request_id" = ?req.id).entered();
        tracing::debug!(?params);
        let result = {
            let _pctx = DbPanicContext::enter(panic_context);
            f(self.global_state, params)
        };
        if let Ok(response) = result_to_response::<R>(req.id, result) {
            self.global_state.respond(response);
        }

        self
    }

    /// Dispatches the request onto the current thread.
    pub(crate) fn on_sync<R>(
        &mut self,
        f: fn(GlobalStateSnapshot, R::Params) -> anyhow::Result<R::Result>,
    ) -> &mut Self
    where
        R: lsp_types::request::Request,
        R::Params: DeserializeOwned + panic::UnwindSafe + fmt::Debug,
        R::Result: Serialize,
    {
        let (req, params, panic_context) = match self.parse::<R>() {
            Some(it) => it,
            None => return self,
        };
        let _guard =
            tracing::info_span!("request", method = ?req.method, "request_id" = ?req.id).entered();
        tracing::debug!(?params);
        let global_state_snapshot = self.global_state.snapshot();

        let result = panic::catch_unwind(move || {
            let _pctx = DbPanicContext::enter(panic_context);
            f(global_state_snapshot, params)
        });

        if let Ok(response) = thread_result_to_response::<R>(req.id, result) {
            self.global_state.respond(response);
        }

        self
    }

    /// Dispatches a non-latency-sensitive request onto the thread pool. When the VFS is marked not
    /// ready this will return a default constructed [`R::Result`].
    pub(crate) fn on<const ALLOW_RETRYING: bool, R>(
        &mut self,
        f: fn(GlobalStateSnapshot, R::Params) -> anyhow::Result<R::Result>,
    ) -> &mut Self
    where
        R: lsp_types::request::Request<
                Params: DeserializeOwned + panic::UnwindSafe + Send + fmt::Debug,
                Result: Serialize + Default,
            > + 'static,
    {
        if !self.global_state.vfs_done {
            if let Some(lsp_server::Request { id, .. }) =
                self.req.take_if(|it| it.method == R::METHOD)
            {
                self.global_state.respond(lsp_server::Response::new_ok(id, R::Result::default()));
            }
            return self;
        }
        self.on_with_thread_intent::<false, ALLOW_RETRYING, R>(
            ThreadIntent::Worker,
            f,
            Self::content_modified_error,
        )
    }

    /// Dispatches a non-latency-sensitive request onto the thread pool. When the VFS is marked not
    /// ready this will return a `default` constructed [`R::Result`].
    pub(crate) fn on_with_vfs_default<R>(
        &mut self,
        f: fn(GlobalStateSnapshot, R::Params) -> anyhow::Result<R::Result>,
        default: impl FnOnce() -> R::Result,
        on_cancelled: fn() -> ResponseError,
    ) -> &mut Self
    where
        R: lsp_types::request::Request<
                Params: DeserializeOwned + panic::UnwindSafe + Send + fmt::Debug,
                Result: Serialize,
            > + 'static,
    {
        if !self.global_state.vfs_done {
            if let Some(lsp_server::Request { id, .. }) =
                self.req.take_if(|it| it.method == R::METHOD)
            {
                self.global_state.respond(lsp_server::Response::new_ok(id, default()));
            }
            return self;
        }
        self.on_with_thread_intent::<false, false, R>(ThreadIntent::Worker, f, on_cancelled)
    }

    /// Dispatches a non-latency-sensitive request onto the thread pool. When the VFS is marked not
    /// ready this will return the parameter as is.
    pub(crate) fn on_identity<const ALLOW_RETRYING: bool, R, Params>(
        &mut self,
        f: fn(GlobalStateSnapshot, Params) -> anyhow::Result<R::Result>,
    ) -> &mut Self
    where
        R: lsp_types::request::Request<Params = Params, Result = Params> + 'static,
        Params: Serialize + DeserializeOwned + panic::UnwindSafe + Send + fmt::Debug,
    {
        if !self.global_state.vfs_done {
            if let Some((request, params, _)) = self.parse::<R>() {
                self.global_state.respond(lsp_server::Response::new_ok(request.id, &params))
            }
            return self;
        }
        self.on_with_thread_intent::<false, ALLOW_RETRYING, R>(
            ThreadIntent::Worker,
            f,
            Self::content_modified_error,
        )
    }

    /// Dispatches a latency-sensitive request onto the thread pool. When the VFS is marked not
    /// ready this will return a default constructed [`R::Result`].
    pub(crate) fn on_latency_sensitive<const ALLOW_RETRYING: bool, R>(
        &mut self,
        f: fn(GlobalStateSnapshot, R::Params) -> anyhow::Result<R::Result>,
    ) -> &mut Self
    where
        R: lsp_types::request::Request<
                Params: DeserializeOwned + panic::UnwindSafe + Send + fmt::Debug,
                Result: Serialize + Default,
            > + 'static,
    {
        if !self.global_state.vfs_done {
            if let Some(lsp_server::Request { id, .. }) =
                self.req.take_if(|it| it.method == R::METHOD)
            {
                self.global_state.respond(lsp_server::Response::new_ok(id, R::Result::default()));
            }
            return self;
        }
        self.on_with_thread_intent::<false, ALLOW_RETRYING, R>(
            ThreadIntent::LatencySensitive,
            f,
            Self::content_modified_error,
        )
    }

    /// Formatting requests should never block on waiting a for task thread to open up, editors will wait
    /// on the response and a late formatting update might mess with the document and user.
    /// We can't run this on the main thread though as we invoke rustfmt which may take arbitrary time to complete!
    pub(crate) fn on_fmt_thread<R>(
        &mut self,
        f: fn(GlobalStateSnapshot, R::Params) -> anyhow::Result<R::Result>,
    ) -> &mut Self
    where
        R: lsp_types::request::Request + 'static,
        R::Params: DeserializeOwned + panic::UnwindSafe + Send + fmt::Debug,
        R::Result: Serialize,
    {
        self.on_with_thread_intent::<true, false, R>(
            ThreadIntent::LatencySensitive,
            f,
            Self::content_modified_error,
        )
    }

    pub(crate) fn finish(&mut self) {
        if let Some(req) = self.req.take() {
            tracing::error!("unknown request: {:?}", req);
            let response = lsp_server::Response::new_err(
                req.id,
                lsp_server::ErrorCode::MethodNotFound as i32,
                "unknown request".to_owned(),
            );
            self.global_state.respond(response);
        }
    }

    fn on_with_thread_intent<const RUSTFMT: bool, const ALLOW_RETRYING: bool, R>(
        &mut self,
        intent: ThreadIntent,
        f: fn(GlobalStateSnapshot, R::Params) -> anyhow::Result<R::Result>,
        on_cancelled: fn() -> ResponseError,
    ) -> &mut Self
    where
        R: lsp_types::request::Request + 'static,
        R::Params: DeserializeOwned + panic::UnwindSafe + Send + fmt::Debug,
        R::Result: Serialize,
    {
        let (req, params, panic_context) = match self.parse::<R>() {
            Some(it) => it,
            None => return self,
        };
        let _guard =
            tracing::info_span!("request", method = ?req.method, "request_id" = ?req.id).entered();
        tracing::debug!(?params);

        let world = self.global_state.snapshot();
        if RUSTFMT {
            &mut self.global_state.fmt_pool.handle
        } else {
            &mut self.global_state.task_pool.handle
        }
        .spawn(intent, move || {
            let result = panic::catch_unwind(move || {
                let _pctx = DbPanicContext::enter(panic_context);
                f(world, params)
            });
            match thread_result_to_response::<R>(req.id.clone(), result) {
                Ok(response) => Task::Response(response),
                Err(_cancelled) if ALLOW_RETRYING => Task::Retry(req),
                Err(_cancelled) => {
                    let error = on_cancelled();
                    Task::Response(Response { id: req.id, result: None, error: Some(error) })
                }
            }
        });

        self
    }

    fn parse<R>(&mut self) -> Option<(lsp_server::Request, R::Params, String)>
    where
        R: lsp_types::request::Request,
        R::Params: DeserializeOwned + fmt::Debug,
    {
        let req = self.req.take_if(|it| it.method == R::METHOD)?;
        let res = crate::from_json(R::METHOD, &req.params);
        match res {
            Ok(params) => {
                let panic_context =
                    format!("\nversion: {}\nrequest: {} {params:#?}", version(), R::METHOD);
                Some((req, params, panic_context))
            }
            Err(err) => {
                let response = lsp_server::Response::new_err(
                    req.id,
                    lsp_server::ErrorCode::InvalidParams as i32,
                    err.to_string(),
                );
                self.global_state.respond(response);
                None
            }
        }
    }

    fn content_modified_error() -> ResponseError {
        ResponseError {
            code: lsp_server::ErrorCode::ContentModified as i32,
            message: "content modified".to_owned(),
            data: None,
        }
    }
}

#[derive(Debug)]
enum HandlerCancelledError {
    Inner(salsa::Cancelled),
}

impl std::error::Error for HandlerCancelledError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HandlerCancelledError::Inner(cancelled) => Some(cancelled),
        }
    }
}

impl fmt::Display for HandlerCancelledError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cancelled")
    }
}

fn thread_result_to_response<R>(
    id: lsp_server::RequestId,
    result: thread::Result<anyhow::Result<R::Result>>,
) -> Result<lsp_server::Response, HandlerCancelledError>
where
    R: lsp_types::request::Request,
    R::Params: DeserializeOwned,
    R::Result: Serialize,
{
    match result {
        Ok(result) => result_to_response::<R>(id, result),
        Err(panic) => {
            let panic_message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied());

            let mut message = "request handler panicked".to_owned();
            if let Some(panic_message) = panic_message {
                message.push_str(": ");
                message.push_str(panic_message);
            } else if let Some(cycle) = panic.downcast_ref::<UnexpectedCycle>() {
                tracing::error!("{cycle}");
                message.push_str(": unexpected cycle");
            } else if let Ok(cancelled) = panic.downcast::<Cancelled>() {
                tracing::error!("Cancellation propagated out of salsa! This is a bug");
                return Err(HandlerCancelledError::Inner(*cancelled));
            };

            Ok(lsp_server::Response::new_err(
                id,
                lsp_server::ErrorCode::InternalError as i32,
                message,
            ))
        }
    }
}

fn result_to_response<R>(
    id: lsp_server::RequestId,
    result: anyhow::Result<R::Result>,
) -> Result<lsp_server::Response, HandlerCancelledError>
where
    R: lsp_types::request::Request,
    R::Params: DeserializeOwned,
    R::Result: Serialize,
{
    let res = match result {
        Ok(resp) => lsp_server::Response::new_ok(id, &resp),
        Err(e) => match e.downcast::<LspError>() {
            Ok(lsp_error) => lsp_server::Response::new_err(id, lsp_error.code, lsp_error.message),
            Err(e) => match e.downcast::<Cancelled>() {
                Ok(cancelled) => return Err(HandlerCancelledError::Inner(cancelled)),
                Err(e) => lsp_server::Response::new_err(
                    id,
                    lsp_server::ErrorCode::InternalError as i32,
                    e.to_string(),
                ),
            },
        },
    };
    Ok(res)
}

pub(crate) struct NotificationDispatcher<'a> {
    pub(crate) not: Option<lsp_server::Notification>,
    pub(crate) global_state: &'a mut GlobalState,
}

impl NotificationDispatcher<'_> {
    pub(crate) fn on_sync_mut<N>(
        &mut self,
        f: fn(&mut GlobalState, N::Params) -> anyhow::Result<()>,
    ) -> &mut Self
    where
        N: lsp_types::notification::Notification,
        N::Params: DeserializeOwned + Send + Debug,
    {
        let not = match self.not.take() {
            Some(it) => it,
            None => return self,
        };

        let _guard = tracing::info_span!("notification", method = ?not.method).entered();

        let params = match not.extract::<N::Params>(N::METHOD) {
            Ok(it) => it,
            Err(ExtractError::JsonError { method, error }) => {
                panic!("Invalid request\nMethod: {method}\n error: {error}",)
            }
            Err(ExtractError::MethodMismatch(not)) => {
                self.not = Some(not);
                return self;
            }
        };

        tracing::debug!(?params);

        let _pctx =
            DbPanicContext::enter(format!("\nversion: {}\nnotification: {}", version(), N::METHOD));
        if let Err(e) = f(self.global_state, params) {
            tracing::error!(handler = %N::METHOD, error = %e, "notification handler failed");
        }
        self
    }

    pub(crate) fn finish(&mut self) {
        if let Some(not) = &self.not {
            if !not.method.starts_with("$/") {
                tracing::error!("unhandled notification: {:?}", not);
            }
        }
    }
}
