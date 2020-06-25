//! A visitor for downcasting arbitrary request (JSON) into a specific type.
use std::panic;

use serde::{de::DeserializeOwned, Serialize};

use crate::{
    global_state::{GlobalState, GlobalStateSnapshot},
    lsp_utils::is_canceled,
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
        R::Params: DeserializeOwned + panic::UnwindSafe + 'static,
        R::Result: Serialize + 'static,
    {
        let (id, params) = match self.parse::<R>() {
            Some(it) => it,
            None => {
                return Ok(self);
            }
        };
        let world = panic::AssertUnwindSafe(&mut *self.global_state);
        let response = panic::catch_unwind(move || {
            let result = f(world.0, params);
            result_to_response::<R>(id, result)
        })
        .map_err(|_| format!("sync task {:?} panicked", R::METHOD))?;
        self.global_state.respond(response);
        Ok(self)
    }

    /// Dispatches the request onto thread pool
    pub(crate) fn on<R>(
        &mut self,
        f: fn(GlobalStateSnapshot, R::Params) -> Result<R::Result>,
    ) -> Result<&mut Self>
    where
        R: lsp_types::request::Request + 'static,
        R::Params: DeserializeOwned + Send + 'static,
        R::Result: Serialize + 'static,
    {
        let (id, params) = match self.parse::<R>() {
            Some(it) => it,
            None => {
                return Ok(self);
            }
        };

        self.global_state.task_pool.handle.spawn({
            let world = self.global_state.snapshot();
            move || {
                let result = f(world, params);
                Task::Response(result_to_response::<R>(id, result))
            }
        });

        Ok(self)
    }

    pub(crate) fn finish(&mut self) {
        if let Some(req) = self.req.take() {
            log::error!("unknown request: {:?}", req);
            let response = lsp_server::Response::new_err(
                req.id,
                lsp_server::ErrorCode::MethodNotFound as i32,
                "unknown request".to_string(),
            );
            self.global_state.respond(response)
        }
    }

    fn parse<R>(&mut self) -> Option<(lsp_server::RequestId, R::Params)>
    where
        R: lsp_types::request::Request + 'static,
        R::Params: DeserializeOwned + 'static,
    {
        let req = self.req.take()?;
        let (id, params) = match req.extract::<R::Params>(R::METHOD) {
            Ok(it) => it,
            Err(req) => {
                self.req = Some(req);
                return None;
            }
        };
        Some((id, params))
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
                if is_canceled(&*e) {
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
