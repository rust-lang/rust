//! A visitor for downcasting arbitrary request (JSON) into a specific type.
use std::{fmt, panic};

use serde::{de::DeserializeOwned, Serialize};

use crate::{
    global_state::{GlobalState, GlobalStateSnapshot},
    lsp_utils::is_cancelled,
    main_loop::Task,
    LspError, Result,
};

pub(crate) struct RequestDispatcher<'a> {
    pub(crate) req: Option<lsp_server::Request>,
    pub(crate) global_state: &'a mut GlobalState,
}

impl<'a> RequestDispatcher<'a> {
    /// Dispatches the request onto the current thread
    pub(crate) fn on_sync<R>(
        &mut self,
        f: fn(&mut GlobalState, R::Params) -> Result<R::Result>,
    ) -> Result<&mut Self>
    where
        R: lsp_types::request::Request + 'static,
        R::Params: DeserializeOwned + panic::UnwindSafe + fmt::Debug + 'static,
        R::Result: Serialize + 'static,
    {
        let (id, params) = match self.parse::<R>() {
            Some(it) => it,
            None => return Ok(self),
        };
        let world = panic::AssertUnwindSafe(&mut *self.global_state);

        let response = panic::catch_unwind(move || {
            let _pctx = stdx::panic_context::enter(format!(
                "\nversion: {}\nrequest: {} {:#?}",
                env!("REV"),
                R::METHOD,
                params
            ));
            let result = f(world.0, params);
            result_to_response::<R>(id, result)
        })
        .map_err(|_err| format!("sync task {:?} panicked", R::METHOD))?;
        self.global_state.respond(response);
        Ok(self)
    }

    /// Dispatches the request onto thread pool
    pub(crate) fn on<R>(
        &mut self,
        f: fn(GlobalStateSnapshot, R::Params) -> Result<R::Result>,
    ) -> &mut Self
    where
        R: lsp_types::request::Request + 'static,
        R::Params: DeserializeOwned + Send + fmt::Debug + 'static,
        R::Result: Serialize + 'static,
    {
        let (id, params) = match self.parse::<R>() {
            Some(it) => it,
            None => return self,
        };

        self.global_state.task_pool.handle.spawn({
            let world = self.global_state.snapshot();

            move || {
                let _pctx = stdx::panic_context::enter(format!(
                    "\nversion: {}\nrequest: {} {:#?}",
                    env!("REV"),
                    R::METHOD,
                    params
                ));
                let result = f(world, params);
                Task::Response(result_to_response::<R>(id, result))
            }
        });

        self
    }

    pub(crate) fn finish(&mut self) {
        if let Some(req) = self.req.take() {
            log::error!("unknown request: {:?}", req);
            let response = lsp_server::Response::new_err(
                req.id,
                lsp_server::ErrorCode::MethodNotFound as i32,
                "unknown request".to_string(),
            );
            self.global_state.respond(response);
        }
    }

    fn parse<R>(&mut self) -> Option<(lsp_server::RequestId, R::Params)>
    where
        R: lsp_types::request::Request + 'static,
        R::Params: DeserializeOwned + 'static,
    {
        let req = match &self.req {
            Some(req) if req.method == R::METHOD => self.req.take().unwrap(),
            _ => return None,
        };

        let res = crate::from_json(R::METHOD, req.params);
        match res {
            Ok(params) => Some((req.id, params)),
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
}

fn result_to_response<R>(
    id: lsp_server::RequestId,
    result: Result<R::Result>,
) -> lsp_server::Response
where
    R: lsp_types::request::Request + 'static,
    R::Params: DeserializeOwned + 'static,
    R::Result: Serialize + 'static,
{
    match result {
        Ok(resp) => lsp_server::Response::new_ok(id, &resp),
        Err(e) => match e.downcast::<LspError>() {
            Ok(lsp_error) => lsp_server::Response::new_err(id, lsp_error.code, lsp_error.message),
            Err(e) => {
                if is_cancelled(&*e) {
                    lsp_server::Response::new_err(
                        id,
                        lsp_server::ErrorCode::ContentModified as i32,
                        "content modified".to_string(),
                    )
                } else {
                    lsp_server::Response::new_err(
                        id,
                        lsp_server::ErrorCode::InternalError as i32,
                        e.to_string(),
                    )
                }
            }
        },
    }
}

pub(crate) struct NotificationDispatcher<'a> {
    pub(crate) not: Option<lsp_server::Notification>,
    pub(crate) global_state: &'a mut GlobalState,
}

impl<'a> NotificationDispatcher<'a> {
    pub(crate) fn on<N>(
        &mut self,
        f: fn(&mut GlobalState, N::Params) -> Result<()>,
    ) -> Result<&mut Self>
    where
        N: lsp_types::notification::Notification + 'static,
        N::Params: DeserializeOwned + Send + 'static,
    {
        let not = match self.not.take() {
            Some(it) => it,
            None => return Ok(self),
        };
        let params = match not.extract::<N::Params>(N::METHOD) {
            Ok(it) => it,
            Err(not) => {
                self.not = Some(not);
                return Ok(self);
            }
        };
        let _pctx = stdx::panic_context::enter(format!(
            "\nversion: {}\nnotification: {}",
            env!("REV"),
            N::METHOD
        ));
        f(self.global_state, params)?;
        Ok(self)
    }

    pub(crate) fn finish(&mut self) {
        if let Some(not) = &self.not {
            if !not.method.starts_with("$/") {
                log::error!("unhandled notification: {:?}", not);
            }
        }
    }
}
